#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <atomic>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <ATen/ThreadLocalState.h>

#include <rice/rice.hpp>
#include <rice/stl.hpp>

#include <ruby/thread.h>
#include <ruby/fiber/scheduler.h>

static_assert(
  TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 15,
  "Incompatible LibTorch version"
);

extern VALUE rb_eTorchError;

extern std::atomic<bool> release_gvl;
extern std::atomic<double> release_gvl_min_flops;
extern std::atomic<int> num_threads_generation;

// grad mode for the current Ruby thread (threads.cpp)
void set_grad_enabled(bool enabled);
void sync_grad_mode();

inline void handle_global_error(const torch::Error& ex) {
  throw Rice::Exception(rb_eTorchError, ex.what_without_backtrace());
}

template<typename T>
void* without_gvl_call(void* data) {
  (*static_cast<T*>(data))();
  return nullptr;
}

// whether any tensor argument is on a device like MPS,
// where ops only queue work and return
inline bool off_cpu(const at::Tensor& t) {
  return t.defined() && !t.device().is_cpu();
}

inline bool off_cpu(const std::optional<at::Tensor>& t) {
  return t && off_cpu(*t);
}

inline bool off_cpu(at::TensorList ts) {
  return std::any_of(ts.begin(), ts.end(), [](const at::Tensor& t) { return off_cpu(t); });
}

inline bool off_cpu(const std::vector<at::Tensor>& ts) {
  return off_cpu(at::TensorList(ts));
}

template<typename T>
bool off_cpu(const T&) {
  return false;
}

// when to release the GVL for an op: below Torch.release_gvl_min_flops,
// the cost of releasing outweighs the gain
namespace gvl_cost {
  // a (..., m, k) times b (..., k, n), where the larger one sets the batch size
  inline double matmul_flops(const at::Tensor& a, const at::Tensor& b) {
    double m = a.dim() >= 2 ? a.size(-2) : 1;
    double n = b.dim() >= 2 ? b.size(-1) : 1;
    return 2 * std::max(a.numel() * n, b.numel() * m);
  }

  // product of the sizes of all indices
  inline double einsum_flops(std::string_view equation, at::TensorList tensors) {
    std::array<int64_t, 128> sizes;
    sizes.fill(-1);
    size_t operand = 0;
    int64_t dim = 0;
    for (char c : equation.substr(0, equation.find("->"))) {
      if (c == ',') {
        operand++;
        dim = 0;
      } else if (c == '.') {
        // assume an ellipsis covers enough work
        return std::numeric_limits<double>::infinity();
      } else if (std::isalpha(static_cast<unsigned char>(c))) {
        if (operand >= tensors.size() || dim >= tensors[operand].dim()) {
          // invalid, so let LibTorch raise
          return 0;
        }
        sizes[c] = tensors[operand].size(dim++);
      }
    }
    double flops = 2;
    for (auto size : sizes) {
      if (size >= 0) {
        flops *= size;
      }
    }
    return flops;
  }

  inline bool large(double flops) {
    return flops >= release_gvl_min_flops.load(std::memory_order_relaxed);
  }

  // always heavy on the CPU, like convolutions and RNNs
  struct heavy {
    template<typename... Args>
    static bool release(const Args&... args) {
      return !(off_cpu(args) || ...);
    }
  };

  // matrix product of the tensor arguments at positions I and J
  template<size_t I, size_t J>
  struct matmul {
    template<typename... Args>
    static bool release(const Args&... args) {
      auto t = std::forward_as_tuple(args...);
      return heavy::release(args...) && large(matmul_flops(std::get<I>(t), std::get<J>(t)));
    }
  };

  // input (..., in) times weight (out, in)
  struct linear {
    template<typename... Args>
    static bool release(const at::Tensor& input, const at::Tensor& weight, const Args&... args) {
      double out = weight.dim() >= 2 ? weight.size(0) : 1;
      return heavy::release(input, weight, args...) && large(2 * input.numel() * out);
    }
  };

  struct einsum {
    template<typename S, typename... Args>
    static bool release(const S& equation, at::TensorList tensors, const Args&...) {
      return !off_cpu(tensors) && large(einsum_flops(std::string_view(equation.data(), equation.size()), tensors));
    }
  };

  // cost doesn't depend on the arguments, like backward and device syncs
  struct always {
    template<typename... Args>
    static bool release(const Args&...) {
      return true;
    }
  };
}

// LibTorch applies Torch.set_num_threads to a native thread only once,
// but M:N threads reuse native threads, so reapply after each change
inline void sync_num_threads() {
  thread_local int seen = 0;
  int generation = num_threads_generation.load(std::memory_order_relaxed);
  if (generation != seen) {
    at::init_num_threads();
    seen = generation;
  }
}

// call f without the GVL when Torch.release_gvl is enabled
// and Cost::release says the op is worth it
// args are converted by the caller while holding the GVL,
// and no Ruby API is used until the GVL is reacquired
// a running op cannot be interrupted (no unblocking function)
template<typename Cost, typename F, typename... Args>
auto without_gvl(F f, Args&&... args) -> std::invoke_result_t<F, Args&&...> {
  using R = std::invoke_result_t<F, Args&&...>;

  sync_num_threads();

  if (!release_gvl.load(std::memory_order_relaxed) || !Cost::release(args...)) {
    return f(std::forward<Args>(args)...);
  }

  int flags = 0;
  // a fiber scheduler may run the call on another native thread,
  // so carry over thread-local state like grad mode
  std::optional<at::ThreadLocalState> state;
  if (rb_fiber_scheduler_current() != Qnil) {
    state.emplace();
#ifdef RB_NOGVL_OFFLOAD_SAFE
    flags |= RB_NOGVL_OFFLOAD_SAFE;
#endif
  }

  std::exception_ptr error;
  std::conditional_t<std::is_void_v<R>, bool, std::optional<R>> result{};
  auto call = [&]() {
    try {
      std::optional<at::ThreadLocalStateGuard> guard;
      if (state) {
        guard.emplace(*state);
        sync_num_threads();
      }
      if constexpr (std::is_void_v<R>) {
        f(std::forward<Args>(args)...);
      } else {
        result.emplace(f(std::forward<Args>(args)...));
      }
    } catch (...) {
      error = std::current_exception();
    }
  };
  rb_nogvl(without_gvl_call<decltype(call)>, &call, nullptr, nullptr, flags);

  if (error) {
    std::rethrow_exception(error);
  }
  if constexpr (!std::is_void_v<R>) {
    return std::move(*result);
  }
}

// keep THP prefix for now to make it easier to compare code

extern VALUE THPDeviceClass;
extern VALUE THPGeneratorClass;
extern VALUE THPVariableClass;

inline VALUE THPUtils_internSymbol(const std::string& str) {
  return Rice::Symbol(str);
}

inline std::string THPUtils_unpackString(VALUE obj) {
  Check_Type(obj, T_STRING);
  return std::string(RSTRING_PTR(obj), RSTRING_LEN(obj));
}

inline bool THPUtils_checkSymbol(VALUE obj) {
  return SYMBOL_P(obj);
}

inline bool THPUtils_checkIndex(VALUE obj) {
  return FIXNUM_P(obj);
}

inline bool THPUtils_checkLong(VALUE obj) {
  return FIXNUM_P(obj);
}

inline bool THPUtils_checkScalar(VALUE obj) {
  return FIXNUM_P(obj) || RB_FLOAT_TYPE_P(obj) || RB_TYPE_P(obj, T_COMPLEX);
}

inline bool THPDevice_Check(VALUE obj) {
  return Rice::detail::protect(rb_obj_is_kind_of, obj, THPDeviceClass);
}

inline bool THPGenerator_Check(VALUE obj) {
  return Rice::detail::protect(rb_obj_is_kind_of, obj, THPGeneratorClass);
}

inline bool THPVariable_Check(VALUE obj) {
  return Rice::detail::protect(rb_obj_is_kind_of, obj, THPVariableClass);
}

inline bool THPVariable_CheckExact(VALUE obj) {
  return Rice::detail::protect(rb_obj_is_instance_of, obj, THPVariableClass);
}
