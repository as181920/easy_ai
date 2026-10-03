#!/usr/bin/env ruby
require "bundler/setup"
require "optparse"
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"
options = { seed: 1337, output: "runs/learning/01_basic_nn/default" }
OptionParser.new do |parser|
  parser.on("--seed N", Integer) { |v| options[:seed] = v }
  parser.on("--output PATH") { |v| options[:output] = v }
end.parse!
EasyAILearning::Course::DataExport.run("01_basic_nn", options)
puts "Saved #{options[:output]}/generated-data.json"
