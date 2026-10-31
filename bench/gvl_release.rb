# ruby bench/gvl_release.rb [scaling|size|serve|device]
# compares holding and releasing the GVL, with 1:1 and M:N threads
# each run is a separate process so OMP_* and RUBY_MN_THREADS take effect

require "etc"
require "rbconfig"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)

CORES = Etc.nprocessors

def measure
  wall = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  cpu = Process.clock_gettime(Process::CLOCK_PROCESS_CPUTIME_ID)
  yield
  [
    Process.clock_gettime(Process::CLOCK_MONOTONIC) - wall,
    Process.clock_gettime(Process::CLOCK_PROCESS_CPUTIME_ID) - cpu
  ]
end

def in_threads(threads, &block)
  threads.times.map { Thread.new(&block) }.each(&:join)
end

def mlp(input, hidden, output)
  Torch::NN::Sequential.new(
    Torch::NN::Linear.new(input, hidden),
    Torch::NN::ReLU.new,
    Torch::NN::Linear.new(hidden, output)
  )
end

# fixed total work (matmuls and training steps) split across threads
def run_scaling(threads)
  iters = 768
  a = Torch.randn(512, 512)
  x = Torch.randn(256, 128)
  y = Torch.randn(256, 1)

  work = lambda do |n|
    model = mlp(128, 512, 1)
    n.times { Torch.matmul(a, a) }
    n.times do
      model.zero_grad
      Torch::NN::Functional.mse_loss(model.call(x), y).backward
    end
  end

  work.call(4)
  measure { in_threads(threads) { work.call(iters / threads) } }
end

# fixed total matmuls of one size split across threads
def run_size(threads, size)
  a = Torch.randn(size, size)
  iters = (2e9 / size**3).clamp(800, 80_000).to_i
  Torch.mm(a, a)
  measure { in_threads(threads) { (iters / threads).times { Torch.mm(a, a) } } }
end

# many threads each running small forward passes, like a server
def run_serve(threads)
  requests = 6400
  model = mlp(64, 256, 10)
  x = Torch.randn(8, 64)
  Torch.no_grad { model.call(x) }
  measure do
    in_threads(threads) do
      Torch.no_grad { (requests / threads).times { model.call(x) } }
    end
  end
end

# fixed total training steps split across threads, on the CPU or MPS
def run_device(threads, device)
  steps = 192
  x = Torch.randn(1024, 512, device: device)
  y = Torch.randn(1024, 1, device: device)

  work = lambda do |n|
    model = mlp(512, 2048, 1).to(device)
    n.times do
      model.zero_grad
      Torch::NN::Functional.mse_loss(model.call(x), y).backward
    end
    model.call(x).sum.item
  end

  work.call(2)
  measure { in_threads(threads) { work.call(steps / threads) } }
end

# training on MPS while another thread runs Ruby code
# returns wall time and how far the Ruby thread got
def run_busy(device)
  x = Torch.randn(1024, 512, device: device)
  y = Torch.randn(1024, 1, device: device)
  model = mlp(512, 2048, 1).to(device)
  count = 0
  ruby = Thread.new { loop { count += 1 } }
  wall, = measure do
    48.times do
      model.zero_grad
      loss = Torch::NN::Functional.mse_loss(model.call(x), y)
      loss.backward
      # read the loss each step, like a training loop that logs it
      loss.item
    end
  end
  ruby.kill.join
  [wall, count]
end

def child(mode, *args, release:, omp:, mn: false, env: {})
  env = env.merge(
    "OMP_NUM_THREADS" => omp.to_s,
    "RUBY_MN_THREADS" => mn ? "1" : "0",
    "TORCH_RB_RELEASE_GVL" => release ? "1" : "0"
  )
  out = IO.popen(env, [RbConfig.ruby, __FILE__, "child", mode, *args.map(&:to_s)], &:read)
  abort "child failed: #{mode} #{args.inspect}" unless $?.success?
  out.split.map(&:to_f)
end

def omp_for(threads)
  [CORES / threads, 1].max
end

def scaling
  puts "fixed work across threads: wall s (speedup vs held)"
  puts "threads  omp     held  released  released+MN"
  [1, 2, 4, 8].each do |threads|
    omp = omp_for(threads)
    held, = child("scaling", threads, release: false, omp: omp)
    released, = child("scaling", threads, release: true, omp: omp)
    mn, = child("scaling", threads, release: true, omp: omp, mn: true)
    puts format("%7d  %3d  %7.3f  %8.3f (%.2fx)  %7.3f (%.2fx)", threads, omp, held, released, held / released, mn, held / mn)
  end
end

def size
  threads = 4
  omp = omp_for(threads)
  puts "Torch.mm on n x n, #{threads} threads, omp #{omp}: us per op"
  puts "   n     held  released  speedup"
  [2, 8, 16, 32, 64, 128, 256].each do |n|
    iters = (2e9 / n**3).clamp(800, 80_000).to_i
    held, = child("size", threads, n, release: false, omp: omp)
    released, = child("size", threads, n, release: true, omp: omp)
    puts format("%4d  %7.2f  %8.2f  %6.2fx", n, held / iters * 1e6, released / iters * 1e6, held / released)
  end
end

def device
  return puts("MPS not available") unless mps?

  puts "training steps on CPU and MPS: wall s"
  puts "device  threads  omp     held  released"
  %w[cpu mps].each do |device|
    [1, 4].each do |threads|
      omp = omp_for(threads)
      held, = child("device", threads, device, release: false, omp: omp)
      released, = child("device", threads, device, release: true, omp: omp)
      puts format("%-6s  %7d  %3d  %7.3f  %8.3f", device, threads, omp, held, released)
    end
  end

  puts
  puts "MPS training (item each step) beside a busy Ruby thread"
  puts "          wall s  Ruby thread iterations"
  [false, true].each do |release|
    wall, count = child("busy", "mps", release: release, omp: CORES)
    puts format("%-8s  %6.3f  %12d", release ? "released" : "held", wall, count)
  end
end

def mps?
  IO.popen({"RUBYLIB" => $LOAD_PATH.first}, [RbConfig.ruby, "-rtorch", "-e", "print Torch::MPS.available?"], &:read) == "true"
end

def serve
  threads = 64
  puts "#{threads} threads of small forward passes: wall s / cpu s"
  puts "config                          held           released"
  [
    ["1:1, omp 1", {omp: 1}],
    ["1:1, omp #{CORES}", {omp: CORES}],
    ["1:1, omp #{CORES}, passive", {omp: CORES, env: {"OMP_WAIT_POLICY" => "passive"}}],
    ["M:N, omp 1", {omp: 1, mn: true}],
    ["M:N, omp #{CORES}", {omp: CORES, mn: true}]
  ].each do |name, opts|
    held = child("serve", threads, release: false, **opts)
    released = child("serve", threads, release: true, **opts)
    puts format("%-28s  %6.3f / %6.3f  %6.3f / %6.3f", name, *held, *released)
  end
end

if ARGV[0] == "child"
  require "torch"
  Torch.manual_seed(1)
  puts send("run_#{ARGV[1]}", *ARGV[2..].map { |v| v.match?(/\A\d+\z/) ? v.to_i : v }).join(" ")
else
  puts "ruby #{RUBY_VERSION}, #{CORES} cores"
  (ARGV.empty? ? %w[scaling size serve device] : ARGV).each do |section|
    puts
    send(section)
  end
end
