require_relative "../test_helper"
require "open3"

class FiberTest < Minitest::Test
  # a timer fiber keeps ticking while other fibers run heavy ops
  SCRIPT = <<~RUBY
    require "torch"
    require "async"

    Torch.release_gvl = true
    a = Torch.randn(3000, 3000)
    x = Torch.randn(3000, 3000, requires_grad: true)
    ticks = 0
    requires_grad = nil
    elapsed = nil

    Sync do |task|
      ticker = task.async { loop { sleep 0.001; ticks += 1 } }
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      4.times.map { task.async { Torch.mm(a, a) } }.each(&:wait)
      # grad mode is carried to the worker thread
      requires_grad = Torch.no_grad { Torch.mm(x, x).requires_grad }
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
      ticker.stop
    end

    puts ticks, (elapsed * 1000).round, requires_grad
  RUBY

  def setup
    skip "Requires Ruby 4.0+" if RUBY_VERSION < "4.0"
  end

  def test_scheduler_offload
    ticks, ideal, requires_grad = run_script("ASYNC_SCHEDULER_WORKER_POOL" => "true")
    assert_operator ticks.to_i, :>, ideal.to_i / 4
    assert_equal "false", requires_grad
  end

  private

  def run_script(env)
    lib = File.expand_path("../../lib", __dir__)
    stdout, stderr, status = Open3.capture3(env, RbConfig.ruby, "-I#{lib}", "-e", SCRIPT)
    assert status.success?, stderr
    stdout.split
  end
end
