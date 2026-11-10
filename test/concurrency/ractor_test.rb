require_relative "../test_helper"
require "json"
require "open3"

class RactorTest < Minitest::Test
  # runs in a separate process, since a Ractor bug can crash the process
  SCRIPT = <<~'RUBY'
    require "torch"
    require "json"

    def value(ractor)
      ractor.respond_to?(:value) ? ractor.value : ractor.take
    end

    def train(seed)
      Torch.manual_seed(seed)
      x = Torch.randn(16, 4)
      y = Torch.randn(16, 1)
      model = Torch::NN::Sequential.new(Torch::NN::Linear.new(4, 8), Torch::NN::ReLU.new, Torch::NN::Linear.new(8, 1))
      optimizer = Torch::Optim::Adam.new(model.parameters, lr: 0.01)
      5.times do
        optimizer.zero_grad
        Torch::NN::Functional.mse_loss(model.call(x), y).backward
        optimizer.step
      end
      Torch.no_grad { Torch::NN::Functional.mse_loss(model.call(x), y).item }
    end

    def ops
      x = Torch.tensor([[1.0, 2.0], [3.0, 4.0]])
      dataset = Torch::Utils::Data::TensorDataset.new(Torch.arange(4))
      {
        "matmul" => Torch.matmul(x, x).to_a,
        "float64" => x.to(:float64).dtype.to_s,
        "long" => Torch::LongTensor.new([1, 2]).to_a,
        "inspect" => x.inspect,
        "no_grad" => Torch.no_grad { (Torch.ones(2, requires_grad: true) * 2).requires_grad },
        "loader" => Torch::Utils::Data::DataLoader.new(dataset, batch_size: 2).map { |(b)| b.to_a }
      }
    end

    begin
      value(Ractor.new { Torch.ones(1) })
    rescue Ractor::RemoteError => e
      raise unless e.cause.is_a?(Ractor::UnsafeError)
      puts JSON.generate("skip" => true)
      exit
    end

    result = {
      "main_ops" => ops,
      "ops" => value(Ractor.new { ops }),
      "main_train" => train(1),
      "train" => value(Ractor.new { train(1) })
    }
    if ENV["PARALLEL"]
      result["parallel"] = 4.times.map { Ractor.new { train(1) } }.map { |r| value(r) }
    end
    puts JSON.generate(result)
  RUBY

  def test_ractor
    result = run_script
    assert_equal result["main_ops"], result["ops"]
    assert_in_delta result["main_train"], result["train"]
  end

  # Rice registers GC roots at runtime, which is only Ractor-safe with Ruby 4.1+
  def test_parallel
    skip "Requires Ruby 4.1+" if RUBY_VERSION < "4.1"

    result = run_script("PARALLEL" => "1")
    result["parallel"].each do |loss|
      assert_in_delta result["main_train"], loss
    end
  end

  private

  def run_script(env = {})
    lib = File.expand_path("../../lib", __dir__)
    stdout, stderr, status = Open3.capture3(env, RbConfig.ruby, "-W0", "-I#{lib}", "-e", SCRIPT)
    assert status.success?, stderr
    result = JSON.parse(stdout)
    skip "Ractors require Ruby 4.1+" if result["skip"]
    result
  end
end
