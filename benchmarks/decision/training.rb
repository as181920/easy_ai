#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "json"
require "fileutils"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
require "easy_ai"

config_path, tokenizer_path, data_path, output, task, steps = ARGV
abort "Usage: ruby benchmarks/decision/training.rb CONFIG TOKENIZER DATA OUTPUT [choice|mlm] [STEPS]" unless output
task = (task || "choice").to_sym
config = EasyAI::Decision::Config.load(config_path).with(training: { steps: Integer(steps || "8") })
tokenizer = EasyAI::Tokenizers::Registry.load(tokenizer_path)
dataset = EasyAI::Decision::Data::Dataset.new(data_path, kind: task)
Torch.manual_seed(config[:training]["seed"])
model = EasyAI::Decision::ChoiceModel.new(config)
trainer = EasyAI::Decision::Trainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: output, task: task)
policy = EasyAI::Runtime::DevicePolicy.new
memories, timings = [], []
started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
gc_started = GC.stat(:time)
previous = started
checkpoint = trainer.train do |_state, _loss|
  now = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  timings << now - previous
  memories << policy.process_memory_mib if trainer.device == "cuda"
  previous = Process.clock_gettime(Process::CLOCK_MONOTONIC)
end
result = { "config" => config.to_h, "task" => task, "parameters" => model.parameter_count,
  "device" => trainer.device, "updates" => trainer.state["step"],
  "update_seconds" => timings, "total_seconds_including_final_checkpoint" => Process.clock_gettime(Process::CLOCK_MONOTONIC) - started,
  "max_sampled_process_gpu_mib" => memories.compact.max, "checkpoint" => checkpoint,
  "gc_seconds" => (GC.stat(:time) - gc_started) / 1000.0,
  "dataset_sha256" => dataset.fingerprint,
  "scope" => "Actual training with tokenization, GC and AdamW; GPU memory sampled after updates, not exact tensor peak. Capacity, not quality benchmark." }
File.write(File.join(output, "benchmark.json"), JSON.pretty_generate(result))
puts JSON.pretty_generate(result)
