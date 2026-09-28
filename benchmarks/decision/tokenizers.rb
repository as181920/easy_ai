#!/usr/bin/env ruby
require "bundler/setup"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
require "easy_ai"
require "benchmark"
require "json"

path = ARGV.fetch(0) { abort "Usage: ruby benchmarks/decision/tokenizers.rb TRAIN.jsonl [VOCAB_SIZE]" }
texts = File.foreach(path).flat_map do |line|
  row = JSON.parse(line)
  row.key?("text") ? [row.fetch("text")] : EasyAI::Decision::Data::Example.new(row).texts
end
vocab_size = Integer(ARGV.fetch(1, "4096"))
results = %w[ruby native].map do |backend|
  tokenizer = EasyAI::Tokenizers::Registry.build(backend)
  training = Benchmark.realtime { tokenizer.train(texts, vocab_size: vocab_size) }
  encoded = nil
  encoding = Benchmark.realtime { encoded = texts.map { |text| tokenizer.encode(text) } }
  round_trip = texts.zip(encoded).all? { |text, ids| tokenizer.decode(ids) == text }
  { "backend" => backend, "documents" => texts.length, "vocab_size" => tokenizer.vocab_size,
    "training_seconds" => training, "encoding_seconds" => encoding,
    "tokens" => encoded.sum(&:length), "round_trip" => round_trip }
end
puts JSON.pretty_generate(results)
