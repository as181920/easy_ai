#!/usr/bin/env ruby
require "optparse"
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"

options = { model: "runs/learning/basic_nn/logic-gates/model.json", device: "auto" }
OptionParser.new do |parser|
  parser.on("--model PATH") { |value| options[:model] = value }
  parser.on("--device NAME", %w[auto cpu cuda]) { |value| options[:device] = value }
end.parse!
model = EasyAILearning::BasicNN::LogicNetwork.load(options[:model], device: options[:device])
puts "Loaded #{options[:model]} on #{model.device}; no training is performed."
puts "x1 x2 | AND OR NAND XOR (scores) | thresholded bits"
EasyAILearning::BasicNN::LogicGates::INPUTS.each do |input|
  scores = model.scores(input)
  puts "#{input.join('  ')}   | #{scores.map { |value| format('% .4f', value) }.join(' ')} | #{scores.map { |value| value >= 0.5 ? 1 : 0 }.join(' ')}"
end
