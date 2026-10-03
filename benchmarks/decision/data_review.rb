#!/usr/bin/env ruby
require_relative "robust"

module DecisionDataReview
  module_function

  def export(source, output)
    raise "Review output exists" if File.exist?(output)
    blind, selected = EasyAI::Decision::Data::QualityAudit.export(DecisionRobust.raw_rows(source))
    FileUtils.mkdir_p(output)
    DecisionRobust.write_rows(File.join(output, "blind.jsonl"), blind)
    DecisionRobust.write_rows(File.join(output, "selected.jsonl"), selected)
    DecisionRobust.write(File.join(output, "manifest.json"), { "source" => File.expand_path(source), "sha256" => DecisionRobust.sha(source),
      "count" => blind.size, "blind_sha256" => DecisionRobust.sha(File.join(output, "blind.jsonl")),
      "selected_sha256" => DecisionRobust.sha(File.join(output, "selected.jsonl")) })
    puts "Exported #{blind.size} target-blind examples to #{output}"
  end

  def import(output, reviews)
    manifest = JSON.parse(File.read(File.join(output, "manifest.json")))
    %w[blind selected].each do |kind|
      raise "Review export changed" unless DecisionRobust.sha(File.join(output, "#{kind}.jsonl")) == manifest.fetch("#{kind}_sha256")
    end
    value = EasyAI::Decision::Data::QualityAudit.adjudicate(DecisionRobust.raw_rows(File.join(output, "selected.jsonl")), DecisionRobust.raw_rows(reviews))
    value["review_file_sha256"] = DecisionRobust.sha(reviews)
    value["reviewers"] = "Codex assistant, single reviewer; not independent human agreement or teacher validation"
    DecisionRobust.write(File.join(output, "audit.json"), value)
    puts JSON.pretty_generate(value.fetch("by_source"))
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "export", source: "runs/decision/factual-v03/pilot/data/candidate-train.jsonl", output: "runs/decision/judgment-v04/review" }
  OptionParser.new do |parser|
    %i[phase source output reviews].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  end.parse!
  case options.fetch(:phase)
  when "export" then DecisionDataReview.export(options.fetch(:source), options.fetch(:output))
  when "import" then DecisionDataReview.import(options.fetch(:output), options.fetch(:reviews))
  else raise ArgumentError, "Unknown review phase"
  end
end
