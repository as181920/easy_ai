#!/usr/bin/env ruby
# Real CUDA regression: full public validation sets, no CPU fallback allowed.
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "tmpdir"
require "json"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
require "easy_ai"

module ValidationMemory
  module_function

  def snapshot(policy)
    rss_kib = File.read("/proc/self/status")[/^VmRSS:\s+(\d+)/, 1].to_i
    { "gpu_process_mib" => policy.process_memory_mib, "rss_mib" => (rss_kib / 1024.0).round(2),
      "ruby_tensor_objects" => ObjectSpace.each_object(Torch::Tensor).count }
  end

  def verify(task, directory, data)
    config = EasyAI::Decision::Config.load("config/decision/semantic.yml")
    config = config.with(training: { device: "cuda", early_stopping_patience: 0 })
    training_file = task == :choice ? "train.jsonl" : "corpus.jsonl"
    validation_file = task == :choice ? "validation.jsonl" : "corpus-validation.jsonl"
    subset = File.join(directory, "#{task}-train.jsonl")
    File.write(subset, File.foreach(File.join(data, training_file)).first(32).join)
    trainer = EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config),
      tokenizer: EasyAI::Tokenizers::Registry.load(File.join(data, "tokenizer.json")),
      task: task, output: File.join(directory, task.to_s),
      dataset: EasyAI::Decision::Data::Dataset.new(subset, kind: task),
      validation: EasyAI::Decision::Data::Dataset.new(File.join(data, validation_file), kind: task))
    trainer.train(steps: 1) # Include gradients and Adam state in the persistent footprint.
    raise "Unexpected CPU fallback" unless trainer.device == "cuda"
    policy = EasyAI::Runtime::DevicePolicy.new
    samples = []
    batch = trainer.method(:validation_batch)
    trainer.define_singleton_method(:validation_batch) do |rows, index|
      # Previous batch was reclaimed by the actual production validation loop.
      samples << ValidationMemory.snapshot(policy).merge("batch" => index) if (index % 8).zero?
      batch.call(rows, index)
    end
    rounds = Array.new(5) do |index|
      before = snapshot(policy)
      loss = trainer.send(:validation_loss)
      after = snapshot(policy)
      raise "Non-finite loss" unless loss.finite?
      raise "Unexpected CPU fallback" unless trainer.device == "cuda"
      $stderr.puts("#{task} validation #{index + 1}/5: GPU #{before['gpu_process_mib']} -> #{after['gpu_process_mib']} MiB; RSS #{after['rss_mib']} MiB")
      { "round" => index + 1, "loss" => loss, "before" => before, "after" => after }
    end
    steady = rounds.drop(2).map { |round| round.fetch("after") }
    ranges = %w[gpu_process_mib rss_mib ruby_tensor_objects].to_h do |key|
      values = steady.map { |row| row.fetch(key) }
      raise "Measurement unavailable: #{key}" if values.any?(&:nil?)
      [key, values.max - values.min]
    end
    raise "CUDA footprint grows after warmup: #{ranges}" if ranges["gpu_process_mib"] > 96
    raise "CPU footprint grows after warmup: #{ranges}" if ranges["rss_mib"] > 128
    raise "Tensor objects retained after validation: #{ranges}" if ranges["ruby_tensor_objects"] > 16
    peak = samples.map { |row| row.fetch("gpu_process_mib") }.compact.max
    raise "GPU budget exceeded" if peak > config[:runtime]["gpu_memory_budget_mib"]
    { "task" => task, "device" => trainer.device, "parameters" => trainer.model.parameter_count,
      "rounds" => rounds, "steady_range" => ranges, "sampled_gpu_peak_mib" => peak, "batch_samples" => samples }
  end
end

abort "CUDA is not accessible" unless Torch::CUDA.available?
data = ARGV.fetch(0, "data/decision/semantic-public")
results = %i[choice mlm].map do |task|
  result = Dir.mktmpdir("easy-ai-validation-memory") { |directory| ValidationMemory.verify(task, directory, data) }
  GC.start
  result
end
puts JSON.pretty_generate("measurement" => "nvidia-smi process footprint (includes allocator/driver caches), Linux RSS, Ruby tensor objects; 2 warmup + 3 measured passes",
  "results" => results)
