require_relative "../test_helper"

class MpsTest < Minitest::Test
  def setup
    skip "MPS not available" unless Torch::MPS.available?

    @previous = [Torch.release_gvl, Torch.release_gvl_min_flops]
    Torch.release_gvl = true
    Torch.release_gvl_min_flops = 0
  end

  def teardown
    Torch.release_gvl, Torch.release_gvl_min_flops = @previous if @previous
  end

  def test_manual_seed
    Torch::MPS.manual_seed(1)
    a = Torch.rand(4, device: "mps").to_a
    Torch::MPS.manual_seed(1)
    assert_equal a, Torch.rand(4, device: "mps").to_a
  end

  def test_synchronize
    a = Torch.randn(1024, 1024, device: "mps")
    Torch.mm(a, a)
    assert_nil Torch::MPS.synchronize
  end

  # ops only queue work, so releasing would only cost time
  def test_queued_ops_hold_gvl
    a = Torch.randn(2048, 2048, device: "mps")
    Torch.mm(a, a).sum.item
    queued = count_during { 20.times { Torch.mm(a, a) } }
    Torch::MPS.synchronize

    assert_operator queued, :<, 10_000
  end

  # getting the GVL back can take a full time slice when another thread is busy
  def test_wait_holds_gvl
    a = Torch.randn(2048, 2048, device: "mps")
    waited = count_during { 10.times.map { Torch.mm(a, a) }.last.sum.item }

    assert_operator waited, :<, 10_000
  end

  def test_synchronize_releases_gvl
    a = Torch.randn(2048, 2048, device: "mps")
    on = count_during do
      10.times { Torch.mm(a, a) }
      Torch::MPS.synchronize
    end

    assert_operator on, :>, 1000
  end

  def test_threads
    Torch.manual_seed(1)
    a = Torch.randn(256, 256)
    x = Torch.randn(64, 32)
    linear = Torch::NN::Linear.new(32, 16)
    expected_mm = Torch.mm(a, a)
    expected_grad = grad(linear, x)

    a_mps = a.to("mps")
    x_mps = x.to("mps")
    linear_mps = Torch::NN::Linear.new(32, 16).to("mps")
    linear_mps.load_state_dict(linear.state_dict)

    results = 8.times.map do
      Thread.new do
        20.times.map { [Torch.mm(a_mps, a_mps).cpu, grad(linear_mps, x_mps).cpu] }
      end
    end.flat_map(&:value)

    results.each do |mm, g|
      assert Torch.allclose(expected_mm, mm, rtol: 1e-4, atol: 1e-4)
      assert Torch.allclose(expected_grad, g, rtol: 1e-4, atol: 1e-4)
    end
  end

  private

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

  def grad(model, x)
    weight = model.weight.detach.clone.requires_grad!
    Torch::NN::Functional.linear(x, weight, model.bias.detach).sum.backward
    weight.grad
  end
end
