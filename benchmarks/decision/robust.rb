#!/usr/bin/env ruby
require_relative "factual"
require_relative "robust/preparation"
require_relative "robust/training"
require_relative "robust/evaluation"
require_relative "robust/report"

# A bounded common-parent comparison; acceptance is opened only after all choices are frozen.
module DecisionRobust
  ROOT = DecisionFactual::ROOT
  PARENT = DecisionFactual::INITIALIZERS.fetch("release")
  CONTROL = File.join(ROOT, "runs/decision/factual-v02-r2")
  ARMS = %w[control candidate].freeze
  STEPS = 2000
  module_function

  def write(path, value)
    DecisionFactual.write(path, value)
  end

  def raw_rows(path)
    DecisionFactual.raw_rows(path)
  end

  def write_rows(path, rows)
    DecisionFactual.write_rows(path, rows)
  end

  def sha(path)
    Digest::SHA256.file(path).hexdigest
  end

  def verify(root)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    protocol.fetch("files_sha256").each do |name, digest|
      raise "Prepared v0.3 data changed: #{name}" unless sha(File.join(root, "data", name)) == digest
    end
    raise "Parent weights changed" unless sha(File.join(PARENT, "weights.pt")) == protocol.fetch("parent_sha256")
    snapshot_path = File.join(root, "baseline-snapshot.json")
    if File.exist?(snapshot_path)
      snapshot = JSON.parse(File.read(snapshot_path))
      raise "Baseline changed after freezing" unless sha(File.join(root, "baseline.json")) == snapshot.fetch("baseline_sha256")
      raise "Protocol changed after baseline" unless sha(File.join(root, "protocol.json")) == snapshot.fetch("protocol_sha256")
    end
    protocol
  end

  def all(root)
    %w[prepare baseline fit].each { |phase| subprocess(root, phase) }
    ARMS.each { |arm| subprocess(root, "train", "--arm", arm) }
    %w[confirm evaluate report].each { |phase| subprocess(root, phase) }
  end

  def subprocess(root, phase, *extra)
    raise "v0.3 #{phase} failed" unless system(RbConfig.ruby, __FILE__, "--phase", phase, "--output", root, *extra)
  end

  def replay(root, source)
    raise "Replay destination exists" if File.exist?(root)
    protocol = verify(source)
    FileUtils.mkdir_p(root)
    FileUtils.cp_r(File.join(source, "data"), File.join(root, "data"))
    protocol["replay_of"] = File.expand_path(source)
    protocol["freshness"] = "All copied panels are previously observed regression material; this replay is not independent acceptance"
    write(File.join(root, "protocol.json"), protocol)
    %w[baseline fit].each { |phase| subprocess(root, phase) }
    ARMS.each { |arm| subprocess(root, "train", "--arm", arm) }
    %w[confirm evaluate report].each { |phase| subprocess(root, phase) }
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/factual-v03/pilot", arm: "candidate", seed: 1337, source: "runs/decision/factual-v03/pilot" }
  OptionParser.new do |parser|
    %i[phase output arm source].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "all", "prepare", "baseline", "fit", "confirm", "evaluate", "report" then DecisionRobust.public_send(options[:phase], root)
  when "train" then DecisionRobust.train(root, options[:arm], options[:seed])
  when "runtime" then DecisionRobust.runtime(root, options[:arm])
  when "replay" then DecisionRobust.replay(root, File.expand_path(options[:source]))
  else raise ArgumentError, "Unknown v0.3 phase"
  end
end
