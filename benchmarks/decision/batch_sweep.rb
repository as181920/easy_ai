#!/usr/bin/env ruby
require "bundler/setup"
require "json"
require "yaml"
require "fileutils"
require "open3"
require "rbconfig"

config_path, tokenizer, data, output = ARGV
abort "Usage: ruby benchmarks/decision/batch_sweep.rb CONFIG TOKENIZER DATA NEW_OUTPUT" unless output
abort "Output already exists: #{output}" if File.exist?(output)
FileUtils.mkdir_p(output)
base = YAML.safe_load_file(config_path)
results = [4, 16, 32].map do |batch|
  config = Marshal.load(Marshal.dump(base))
  config.fetch("training").merge!("choice_microbatch" => batch, "gradient_accumulation" => 32 / batch, "early_stopping_patience" => 0)
  path = File.join(output, "batch-#{batch}.yml")
  File.write(path, YAML.dump(config))
  run = File.join(output, "batch-#{batch}")
  stdout, stderr, status = Open3.capture3(RbConfig.ruby, File.join(__dir__, "training.rb"), path, tokenizer, data, run, "choice", "12")
  abort stderr unless status.success?
  result = JSON.parse(stdout)
  times = result.fetch("update_seconds").drop(2)
  mean = times.sum / times.length
  summary = { "microbatch" => batch, "accumulation" => 32 / batch, "device" => result.fetch("device"),
    "seconds_per_update" => mean, "examples_per_second" => 32 / mean,
    "sampled_process_gpu_mib" => result["max_sampled_process_gpu_mib"], "gc_seconds" => result.fetch("gc_seconds") }
  warn JSON.generate(summary)
  summary
end
File.write(File.join(output, "batch-sweep.json"), JSON.pretty_generate(results))
puts JSON.pretty_generate(results)
