#!/usr/bin/env ruby
require "optparse"
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"

options = { device: "auto", seed: 1337, steps: 10_000, learning_rate: 0.1, output: "runs/learning/basic_nn/logic-gates" }
OptionParser.new do |parser|
  parser.banner = "Usage: bundle exec ruby learning/01_basic_nn/train.rb [options]"
  parser.on("--device NAME", %w[auto cpu cuda]) { |value| options[:device] = value }
  parser.on("--seed N", Integer) { |value| options[:seed] = value }
  parser.on("--steps N", Integer) { |value| options[:steps] = value }
  parser.on("--learning-rate N", Float) { |value| options[:learning_rate] = value }
  parser.on("--output PATH") { |value| options[:output] = value }
end.parse!

model = EasyAILearning::BasicNN::LogicNetwork.new(seed: options[:seed], device: options[:device])
trainer = EasyAILearning::BasicNN::LogicTrainer.new(model: model, learning_rate: options[:learning_rate], max_steps: options[:steps])
puts "Torch.rb device=#{model.device}: 2 -> 2 ReLU -> 1 linear (XOR); #{model.parameter_count} trainable parameters."
puts "Full-batch gradient descent over all four input combinations; all coefficients are trained."
trainer.train { |step, loss| puts format("step=%d mse=%.8f", step, loss) if step == 1 || (step % 200).zero? }
puts "steps=#{trainer.steps} converged=#{trainer.converged?} max_error=#{trainer.max_error.round(6)}"
report = EasyAILearning::BasicNN::LogicReport.new(trainer: trainer, seed: options[:seed])
puts report.table
puts "\nTrained parameters (rounded for display; full precision in model.json):"
puts report.equations
puts report.write(options[:output])
puts "\nSaved model.json, loss.json, plots.txt in #{File.expand_path(options[:output])}"
puts "Intermediate real inputs were not trained; slices illustrate the learned continuous function, not Boolean labels."
abort "Training did not meet max-error <= 0.01; inspect the result instead of treating it as a success." unless trainer.converged?
