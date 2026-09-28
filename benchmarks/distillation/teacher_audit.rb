#!/usr/bin/env ruby
ENV["OMP_NUM_THREADS"] ||= "1"
ENV["MKL_NUM_THREADS"] ||= "1"
require "bundler/setup"
require "optparse"
require "yaml"
require_relative "../../lib/easy_ai"

# A small, fixed development diagnostic. Never reads test or creates student labels.
options = { semantic_data: "data/decision/semantic-public", relation_data: "data/decision/relations-v3" }
OptionParser.new do |parser|
  %i[teacher_config output semantic_data relation_data prompt_profile].each do |key|
    parser.on("--#{key.to_s.tr('_', '-')} PATH") { |value| options[key] = value }
  end
  parser.on("--prepare-only") { options[:prepare_only] = true }
end.parse!
output = options.fetch(:output)
data_path = File.join(output, "development.jsonl")
sources = { "public" => File.join(options[:semantic_data], "validation.jsonl"),
  "relations" => File.join(options[:relation_data], "validation.jsonl") }
protocol = { "version" => 1, "scope" => "Small development diagnostic; not a blind benchmark or teacher quality guarantee",
  "source_sha256" => sources.transform_values { |path| Digest::SHA256.file(path).hexdigest },
  "selection" => "First four distinct groups per public source/label; first two complete mixed-truth binding groups per language",
  "gate" => { "relation_accuracy" => 0.95, "public_source_macro_accuracy" => 0.70, "permutation_agreement" => 0.95 } }
FileUtils.mkdir_p(output)
protocol_path = File.join(output, "protocol.json")
if File.exist?(protocol_path)
  raise ArgumentError, "Audit inputs changed" unless JSON.parse(File.read(protocol_path)) == protocol
else
  EasyAI::Distillation::Artifact.write_json(protocol_path, protocol)
end
rows = []
seen = Hash.new { |hash, key| hash[key] = Set.new }
EasyAI::Decision::Data::Dataset.new(sources.fetch("public")).each do |row|
  groups = seen[[row.source, row.target]]
  next if groups.size >= 4 || groups.include?(row.group_id)
  groups << row.group_id
  rows << row.to_h
end
relations = File.foreach(sources.fetch("relations")).map { |line| JSON.parse(line) }
relations.group_by { |row| row.fetch("language") }.each_value do |language_rows|
  mixed = language_rows.select { |row| row.fetch("relation").fetch("facts").uniq.size == 2 }
  mixed.group_by { |row| row.fetch("relation").fetch("binding_group") }.first(2).each do |_key, group|
    group.each { |row| rows << EasyAI::Decision::Data::Example.new(row).to_h }
  end
end
prepared = rows.flat_map do |row|
  original = row.merge("id" => "original:#{row.fetch('id')}")
  [original, row.merge("id" => "reversed:#{row.fetch('id')}", "options" => row.fetch("options").reverse)]
end
contents = prepared.map { |row| JSON.generate(row) }.join("\n") + "\n"
raise ArgumentError, "Audit selection changed" if File.exist?(data_path) && File.read(data_path) != contents
File.write(data_path, contents) unless File.exist?(data_path)
data = EasyAI::Decision::Data::Dataset.new(data_path)
if options[:prepare_only]
  puts JSON.pretty_generate(protocol.merge("rows" => data.size, "data_sha256" => data.fingerprint))
  exit
end
config = YAML.safe_load_file(options.fetch(:teacher_config), aliases: false)
teacher = EasyAI::Distillation::Teachers::LocalHttp.new(**config.transform_keys(&:to_sym))
adapter = EasyAI::Decision::DistillationAdapter.new(profile: options.fetch(:prompt_profile, "generic"))
artifact = EasyAI::Distillation::Collector.new(teacher: teacher, adapter: adapter,
  output: File.join(output, "teacher"), purpose: "development").run(data)
results = data.map do |row|
  label = artifact.fetch(row.id).fetch("supervision").fetch("target")
  { "id" => row.id, "source" => row.source, "language" => row.language, "target" => row.target, "prediction" => label,
    "correct" => label == row.target }
end
original = results.select { |row| row.fetch("id").start_with?("original:") }
by_source = original.group_by { |row| row.fetch("source") }.transform_values do |group|
  { "count" => group.size, "accuracy" => group.count { |row| row.fetch("correct") }.fdiv(group.size) }
end
indexed = results.to_h { |row| [row.fetch("id"), row] }
agreement = original.count do |row|
  row.fetch("prediction") == indexed.fetch(row.fetch("id").sub("original:", "reversed:")).fetch("prediction")
end.fdiv(original.size)
public_metrics = by_source.reject { |source, _| source == "relations" }
relation_metrics = by_source.fetch("relations")
macro = public_metrics.values.sum { |value| value.fetch("accuracy") } / public_metrics.size
summary = protocol.merge("artifact_sha256" => artifact.fingerprint, "by_source" => by_source,
  "by_language" => original.group_by { |row| row.fetch("language") }.transform_values { |group| group.count { |row| row.fetch("correct") }.fdiv(group.size) },
  "public_source_macro_accuracy" => macro, "permutation_agreement" => agreement,
  "passed" => relation_metrics.fetch("accuracy") >= 0.95 && macro >= 0.70 && agreement >= 0.95,
  "rows" => results)
EasyAI::Distillation::Artifact.write_json(File.join(output, "summary.json"), summary)
puts JSON.pretty_generate(summary.reject { |key, _| key == "rows" })
