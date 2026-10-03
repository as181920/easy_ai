#!/usr/bin/env ruby
require "json"
require "optparse"
options = { model: "runs/learning/00_foundations/default/linear-model.json", value: 0.25 }
OptionParser.new do |p|
  p.on("--model PATH") { |v| options[:model] = v }
  p.on("--value N", Float) { |v| options[:value] = v }
end.parse!
model = JSON.parse(File.read(options[:model]))
puts JSON.pretty_generate(input: options[:value], output: model.fetch("weights").first * options[:value] + model.fetch("bias"))
