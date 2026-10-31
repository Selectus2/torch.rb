#include <torch/torch.h>
#include <torch/mps.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_mps(Rice::Module& m) {
  Rice::define_module_under(m, "MPS")
    .define_singleton_function("available?", &torch::mps::is_available)
    .define_singleton_function("manual_seed", &torch::mps::manual_seed)
    .define_singleton_function(
      "synchronize",
      []() {
        // waits for queued work, so other threads can run meanwhile
        // (other waits like Tensor#item hold the GVL, since getting it back
        // can take a full time slice when another thread is busy)
        without_gvl<gvl_cost::always>([]() { torch::mps::synchronize(); });
      });
}
