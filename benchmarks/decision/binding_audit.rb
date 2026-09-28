#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "optparse"
require_relative "../../lib/easy_ai"

# Development diagnostics, deliberately separate from blind test evaluation.
options = { data: "data/decision/relations-v3", device: "auto", batch_size: 128 }
OptionParser.new do |parser|
  %i[checkpoint data output device].each { |key| parser.on("--#{key} PATH") { |value| options[key] = value } }
  parser.on("--batch-size N", Integer) { |value| options[:batch_size] = value }
end.parse!
checkpoint = options.fetch(:checkpoint)
output = options.fetch(:output)
raise ArgumentError, "Output exists: #{output}" if File.exist?(output)
data = options.fetch(:data)
manifest = JSON.parse(File.read(File.join(data, "manifest.json")))
raise ArgumentError, "Expected v3 relation metadata" unless manifest["version"] == 3
manifest.fetch("files_sha256").each do |name, checksum|
  raise ArgumentError, "Dataset changed: #{name}" unless Digest::SHA256.file(File.join(data, name)).hexdigest == checksum
end
resolved = EasyAI::Decision::Checkpoint.resolve(checkpoint)
metadata = JSON.parse(File.read(File.join(resolved, "metadata.json")))
raise ArgumentError, "Tokenizer differs from audit dataset" unless metadata["tokenizer_fingerprint"] == manifest["tokenizer_fingerprint"]
predictor = EasyAI::Decision::Predictor.load(checkpoint, device: options[:device])
evaluator = EasyAI::Decision::RelationEvaluation.new(predictor, batch_size: options[:batch_size])
FileUtils.mkdir_p(output)
summary = { "checkpoint" => resolved, "weights_sha256" => Digest::SHA256.file(File.join(resolved, "weights.pt")).hexdigest,
  "data_manifest" => manifest, "config" => predictor.model.config.to_h,
  "scope" => "Training/validation and known user probes; not blind generalization evidence", "splits" => {} }
%w[train validation].each do |split|
  warn "Auditing #{split} on #{predictor.device}"
  result = evaluator.evaluate(EasyAI::Decision::Data::Dataset.new(File.join(data, "#{split}.jsonl")))
  File.write(File.join(output, "#{split}.json"), JSON.pretty_generate(result))
  summary["splits"][split] = result.slice("accuracy", "pairs", "groups", "by_language", "by_fact_pattern", "device")
end
probes = [
  ["小林买了票，小周没有买票。", "以下说法成立吗：小周买了票。", "成立", "不成立", 1],
  ["小周买了票，小林没有买票。", "以下说法成立吗：小周买了票。", "成立", "不成立", 0],
  ["我要迟到啦！", "是不是要迟到了？", "是", "不是", 0],
  ["还早，不会迟到啦。", "是不是要迟到了？", "是", "不是", 1],
  ["I think i will be late", "will I late?", "will", "no", 0],
  ["Time is enough , I will not  be late", "will I late?", "will", "no", 1]
]
summary["known_probes"] = probes.map do |state, question, yes, no, expected|
  request = { state: state, question: question, options: [{ id: 0, text: yes }, { id: 1, text: no }] }
  first = predictor.probabilities(**request)
  predictor.clear_cache
  second = predictor.probabilities(**request)
  probabilities = first.fetch("probabilities")
  { "request" => request, "result" => first, "expected_text_direction" => expected.to_s,
    "correct" => probabilities.max_by { |_id, probability| probability }.first == expected.to_s,
    "cache_clear_maximum_difference" => probabilities.map { |id, p| (p - second["probabilities"][id]).abs }.max }
end
File.write(File.join(output, "summary.json"), JSON.pretty_generate(summary))
puts JSON.pretty_generate(summary.slice("checkpoint", "scope", "known_probes"))
