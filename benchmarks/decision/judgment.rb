#!/usr/bin/env ruby
require_relative "robust"
require_relative "judgment/preparation"
require_relative "judgment/training"
require_relative "judgment/evaluation"
require_relative "judgment/report"

module DecisionJudgment
  PARENT = DecisionRobust::PARENT
  ARMS = %w[control candidate].freeze
  module_function

  def write(path, value)
    DecisionRobust.write(path, value)
  end

  def raw(path)
    DecisionRobust.raw_rows(path)
  end

  def sha(path)
    DecisionRobust.sha(path)
  end

  def verify(root)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    protocol.fetch("files_sha256").each do |name, digest|
      raise "Frozen data changed: #{name}" unless sha(File.join(root, "data", name)) == digest
    end
    raise "Parent changed" unless sha(File.join(PARENT, "weights.pt")) == protocol.fetch("parent_sha256")
    snapshot = File.join(root, "baseline-snapshot.json")
    if File.exist?(snapshot)
      value = JSON.parse(File.read(snapshot))
      raise "Protocol changed" unless value.fetch("protocol_sha256") == sha(File.join(root, "protocol.json"))
      raise "Baseline changed" unless value.fetch("baseline_sha256") == sha(File.join(root, "baseline.json"))
    end
    protocol
  end

  def subprocess(root, phase, *extra)
    raise "Judgment #{phase} failed" unless system(RbConfig.ruby, __FILE__, "--output", root, "--phase", phase, *extra)
  end

  def all(root)
    subprocess(root, "prepare")
    run_prepared(root)
  end

  def run_prepared(root)
    %w[baseline fit].each { |phase| subprocess(root, phase) }
    # A failed bounded fitting check ends the round, rather than training on a broken path.
    unless JSON.parse(File.read(File.join(root, "fit/result.json"))).fetch("passed")
      subprocess(root, "report")
      return
    end
    ARMS.each { |arm| subprocess(root, "train", "--arm", arm) }
    %w[confirm evaluate report].each { |phase| subprocess(root, phase) }
  end

  def replay(root, source)
    raise "Replay destination exists" if File.exist?(root)
    protocol = verify(source)
    FileUtils.mkdir_p(root)
    FileUtils.cp_r(File.join(source, "data"), File.join(root, "data"))
    protocol["replay_of"] = File.expand_path(source)
    protocol["scope"] += " Replay: copied panels are regression material and cannot support new promotion."
    write(File.join(root, "protocol.json"), protocol)
    run_prepared(root)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/judgment-v04/pilot", source: "runs/decision/judgment-v04/pilot", arm: "candidate", seed: 1337 }
  OptionParser.new do |parser|
    %i[phase output source arm].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options.fetch(:output))
  case options.fetch(:phase)
  when "all", "prepare", "baseline", "fit", "confirm", "evaluate", "report" then DecisionJudgment.public_send(options.fetch(:phase), root)
  when "train" then DecisionJudgment.train(root, options.fetch(:arm), options.fetch(:seed))
  when "runtime" then DecisionJudgment.runtime(root, options.fetch(:arm))
  when "replay" then DecisionJudgment.replay(root, File.expand_path(options.fetch(:source)))
  else raise ArgumentError, "Unknown judgment phase"
  end
end
