require_relative "../test_helper"
require "open3"

class GradModeTest < Minitest::Test
  # more threads than RUBY_MAX_CPU (8) so M:N threads share native threads
  SCRIPT = <<~RUBY
    require "torch"

    errors = 32.times.map do |i|
      Thread.new do
        expected = i.even?
        Torch.grad_enabled(expected) do
          x = Torch.ones(2, requires_grad: true)
          index = Torch.tensor([0])
          # each check runs first after a thread switch
          checks = [
            -> { x[index].requires_grad },
            -> { x.to(:float64).requires_grad },
            -> { (x * 2).requires_grad },
            -> { Torch.grad_enabled? }
          ]
          200.times.count do |j|
            Thread.pass
            checks[j % checks.size].call != expected
          end
        end
      end
    end.sum(&:value)
    # threads that exit with grad mode disabled don't affect new threads
    8.times.map { Thread.new { Torch._set_grad_enabled(false) } }.each(&:join)
    errors += 8.times.map { Thread.new { Torch.grad_enabled? ? 0 : 1 } }.sum(&:value)
    puts errors
  RUBY

  def test_threads
    assert_equal 0, run_script(mn: false)
  end

  def test_mn_threads
    assert_equal 0, run_script(mn: true)
  end

  private

  def run_script(mn:)
    env = {"RUBY_MN_THREADS" => mn ? "1" : "0"}
    lib = File.expand_path("../../lib", __dir__)
    stdout, stderr, status = Open3.capture3(env, RbConfig.ruby, "-I#{lib}", "-e", SCRIPT)
    assert status.success?, stderr
    stdout.to_i
  end
end
