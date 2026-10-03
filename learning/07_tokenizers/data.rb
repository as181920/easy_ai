#!/usr/bin/env ruby
require "bundler/setup"
require "optparse"
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"
options = { seed: 1337, output: "runs/learning/07_tokenizers/default" }
OptionParser.new do |parser|
  parser.on("--seed N", Integer) { |v| options[:seed] = v }
  parser.on("--output PATH") { |v| options[:output] = v }
end.parse!
EasyAILearning::Course::DataExport.run("07_tokenizers", options)
puts "Saved #{options[:output]}/generated-data.json"
