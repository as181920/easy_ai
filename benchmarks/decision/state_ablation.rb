ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "json"
require_relative "../../lib/easy_ai"

root = File.expand_path(ARGV.fetch(0) { abort "Usage: ruby benchmarks/decision/state_ablation.rb PIPELINE_RUN [LANGUAGE] [CHECKPOINT]" })
summary = JSON.parse(File.read(File.join(root, "summary.json")))
data = summary.fetch("data_directory", File.join(root, "data"))
checkpoint = ARGV[2] || summary.dig("selection", "choice") || File.join(root, "choice")
predictor = EasyAI::Decision::Predictor.load(checkpoint, device: "cpu")
report = EasyAI::Decision::StateAblation.new(predictor: predictor,
  validation: EasyAI::Decision::Data::Dataset.new(File.join(data, "validation.jsonl")),
  reference: EasyAI::Decision::Data::Dataset.new(File.join(data, "train.jsonl")), language: ARGV[1]).evaluate
report["checkpoint"] = checkpoint
File.write(File.join(root, "state-ablation.json"), JSON.pretty_generate(report))
puts JSON.pretty_generate(report)
