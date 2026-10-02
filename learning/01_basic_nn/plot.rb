#!/usr/bin/env ruby
require "json"
require "optparse"
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"

options = { run: "runs/learning/basic_nn/logic-gates", output: "runs/learning/basic_nn/logic-gates/figure" }
OptionParser.new do |parser|
  parser.on("--run PATH") { |value| options[:run] = value }
  parser.on("--output PATH") { |value| options[:output] = value }
end.parse!
metadata = JSON.parse(File.read(File.join(options[:run], "model.json")))
history = JSON.parse(File.read(File.join(options[:run], "loss.json")))
model = EasyAILearning::BasicNN::LogicNetwork.load(File.join(options[:run], "model.json"), device: :cpu)
figure = EasyAILearning::BasicNN::LogicFigure.new(model: model, history: history, metadata: metadata)
puts figure.write(options[:output])
