require_relative "../test_helper"
require "open3"

class GvlReleaseTest < Minitest::Test
  def setup
    @previous = [Torch.release_gvl, Torch.release_gvl_min_flops]
    Torch.release_gvl = true
    Torch.release_gvl_min_flops = 0
  end

  def teardown
    Torch.release_gvl, Torch.release_gvl_min_flops = @previous
  end

  def test_switch
    Torch.release_gvl = false
    refute Torch.release_gvl
    Torch.release_gvl = true
    assert Torch.release_gvl
  end

  def test_threads_matmul
    Torch.manual_seed(1)
    inputs = 4.times.map { [Torch.randn(64, 32), Torch.randn(32, 16)] }
    expected = inputs.map { |a, b| Torch.matmul(a, b) }

    results = inputs.map { |a, b| Thread.new { 20.times.map { Torch.matmul(a, b) } } }.map(&:value)
    expected.zip(results) do |exp, res|
      res.each { |r| assert Torch.equal(exp, r) }
    end
  end

  def test_threads_conv2d
    Torch.manual_seed(1)
    conv = Torch::NN::Conv2d.new(3, 8, 3)
    inputs = 4.times.map { Torch.randn(2, 3, 16, 16) }
    expected = Torch.no_grad { inputs.map { |x| conv.call(x) } }

    results = inputs.map { |x| Thread.new { Torch.no_grad { conv.call(x) } } }.map(&:value)
    expected.zip(results) do |exp, res|
      assert Torch.equal(exp, res)
    end
  end

  def test_releases_gvl
    a = Torch.randn(3000, 3000)
    on = count_during { Torch.mm(a, a) }

    Torch.release_gvl = false
    off = count_during { Torch.mm(a, a) }

    assert_operator on, :>, 1000
    assert_operator off, :<, on / 10
  end

  def test_min_flops_mm
    a = Torch.randn(3000, 3000)
    assert_min_flops(2 * 3000**3) { Torch.mm(a, a) }
  end

  def test_min_flops_matmul_broadcast
    a = Torch.randn(4, 1000, 1000)
    b = Torch.randn(1000, 1000)
    assert_min_flops(2 * 4 * 1000**3) { Torch.matmul(a, b) }
  end

  def test_min_flops_linear
    x = Torch.randn(3000, 2000)
    weight = Torch.randn(1000, 2000)
    assert_min_flops(2 * 3000 * 2000 * 1000) { Torch::NN::Functional.linear(x, weight, nil) }
  end

  def test_min_flops_einsum
    a = Torch.randn(3000, 2000)
    b = Torch.randn(2000, 1000)
    assert_min_flops(2 * 3000 * 2000 * 1000) { Torch.einsum("ij,jk->ik", [a, b]) }
  end

  def test_num_threads
    previous = Torch.get_num_threads
    begin
      Torch.set_num_threads(2)
      assert_equal 2, Torch.get_num_threads
      assert_equal 2, Thread.new { Torch.mm(Torch.ones(2, 2), Torch.ones(2, 2)); Torch.get_num_threads }.value
    ensure
      Torch.set_num_threads(previous)
    end
  end

  def test_backward_ignores_min_flops
    Torch.release_gvl_min_flops = Float::INFINITY
    a = Torch.randn(2000, 2000, requires_grad: true)
    loss = Torch.mm(a, a).sum
    released = count_during { loss.backward }

    Torch.release_gvl = false
    loss = Torch.mm(a, a).sum
    held = count_during { loss.backward }

    assert_operator held, :<, released / 10
  end

  def test_error
    error = assert_raises(Torch::Error) do
      Torch.mm(Torch.ones(2, 3), Torch.ones(2, 3))
    end
    assert_match "cannot be multiplied", error.message
  end

  def test_error_in_thread
    error = assert_raises(Torch::Error) do
      thread = Thread.new { Torch.matmul(Torch.ones(2, 3), Torch.ones(2, 3)) }
      thread.report_on_exception = false
      thread.join
    end
    assert_match "cannot be multiplied", error.message
  end

  def test_no_grad
    a = Torch.ones(2, 2, requires_grad: true)
    assert Torch.mm(a, a).requires_grad
    refute Torch.no_grad { Torch.mm(a, a) }.requires_grad
  end

  def test_threads_backward
    Torch.manual_seed(1)
    model = Torch::NN::Linear.new(8, 1)
    inputs = 4.times.map { Torch.randn(16, 8) }
    expected = inputs.map { |x| grad(model, x) }

    results = inputs.map { |x| Thread.new { grad(model, x) } }.map(&:value)
    expected.zip(results) do |exp, res|
      assert Torch.allclose(exp, res)
    end
  end

  # libomp crashed when M:N threads ran parallel ops on several native threads
  def test_mn_threads_openmp
    script = <<~RUBY
      require "torch"
      Torch.release_gvl = true
      Torch.release_gvl_min_flops = 0
      a = Torch.randn(256, 256)
      x = Torch.randn(256, 128)
      8.times.map do
        Thread.new do
          l1 = Torch::NN::Linear.new(128, 512)
          l2 = Torch::NN::Linear.new(512, 1)
          100.times do
            Torch.matmul(a, a)
            l2.call(l1.call(x).relu).sum.backward
          end
        end
      end.each(&:join)
    RUBY
    env = {"RUBY_MN_THREADS" => "1", "OMP_NUM_THREADS" => "4"}
    lib = File.expand_path("../../lib", __dir__)
    _, stderr, status = Open3.capture3(env, RbConfig.ruby, "-I#{lib}", "-e", script)
    assert status.success?, stderr.lines.first(5).join
  end

  private

  # count iterations of a Ruby loop in another thread while the block runs
  def count_during
    count = 0
    counter = Thread.new { loop { count += 1 } }
    sleep(0.05) until count > 0
    # start with a fresh time slice
    Thread.pass
    before = count
    yield
    after = count
    counter.kill.join
    after - before
  end

  def assert_min_flops(flops, &block)
    Torch.release_gvl_min_flops = flops + 1
    held = count_during(&block)

    Torch.release_gvl_min_flops = flops
    released = count_during(&block)

    assert_operator held, :<, released / 10
  end

  def grad(model, x)
    weight = model.weight.detach.clone.requires_grad!
    Torch::NN::Functional.linear(x, weight, model.bias.detach).sum.backward
    weight.grad
  end
end
