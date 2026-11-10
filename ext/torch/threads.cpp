#include <cstdint>

#include <torch/torch.h>

#include <rice/rice.hpp>

#include <ruby/thread.h>

#include "utils.h"

// LibTorch keeps grad mode per native thread, but Ruby threads can move
// between native threads (M:N threads, which Ractors always use)
// so keep grad mode per Ruby thread, and apply it before each op
// fibers share the grad mode of their thread

namespace {
  enum : intptr_t { GRAD_MODE_UNSET = 0, GRAD_MODE_DISABLED, GRAD_MODE_ENABLED };

  rb_internal_thread_specific_key_t grad_mode_key;
}

void sync_grad_mode() {
  auto mode = reinterpret_cast<intptr_t>(rb_internal_thread_specific_get(rb_thread_current(), grad_mode_key));
  // new threads start with grad mode enabled, like native threads
  bool enabled = mode != GRAD_MODE_DISABLED;
  if (torch::GradMode::is_enabled() != enabled) {
    torch::GradMode::set_enabled(enabled);
  }
}

void set_grad_enabled(bool enabled) {
  auto mode = enabled ? GRAD_MODE_ENABLED : GRAD_MODE_DISABLED;
  rb_internal_thread_specific_set(rb_thread_current(), grad_mode_key, reinterpret_cast<void*>(mode));
  torch::GradMode::set_enabled(enabled);
}

void init_threads(Rice::Module& m) {
  grad_mode_key = rb_internal_thread_specific_key_create();
}
