#!/usr/bin/env ruby
require "bundler/setup"
require "optparse"
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "easy_ai_learning"
options = { model: "runs/learning/07_tokenizers/default/bpe-tokenizer.json", text: "你好 world\n" }
OptionParser.new do |p|
  p.on("--model PATH") { |v| options[:model] = v }
  p.on("--text TEXT") { |v| options[:text] = v }
end.parse!
model = EasyAILearning::Tokenizers::ReversibleBpe.load(options[:model])
ids = model.encode(options[:text])
puts JSON.pretty_generate(input: options[:text], ids: ids, decoded: model.decode(ids))
