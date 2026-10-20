#!/usr/bin/env ruby
require_relative "binding_generalization"
require_relative "event_generalization/catalog"
require_relative "event_generalization/corpus"
require_relative "event_generalization/augmentation"
require_relative "event_generalization/trainer"
require_relative "event_generalization/policy"
require_relative "event_generalization/protocol"
require_relative "event_generalization/experiment"
require_relative "event_generalization/audit"
require_relative "event_generalization/report"

module DecisionEventGeneralization
  ROOT_V07 = File.expand_path("runs/decision/generalization-v07")
  VARIANTS = %w[reference coverage event combined].freeze
  CEILING = 37_000
  module_function

  def write(path, value)
    FileUtils.mkdir_p(File.dirname(path))
    DecisionScaling.write(path, value)
  end

  def sha(path)
    DecisionScaling.sha(path)
  end

  def replay(root, source)
    raise "Replay destination exists" if File.exist?(root)
    verify(source)
    FileUtils.mkdir_p(root)
    %w[data forbidden.json protocol.json protocol.json.sha256.json plan-frozen.md source-frozen isolation-correction].each do |name|
      FileUtils.cp_r(File.join(source, name), File.join(root, name))
    end
    FileUtils.mkdir_p(File.join(root, "preflight"))
    FileUtils.cp_r(File.join(source, "preflight/inherited"), File.join(root, "preflight/inherited"))
    seal(File.join(root, "replay-marker.json"), { "source" => source, "protocol_sha256" => sha(File.join(source, "protocol.json")),
      "scope" => "Copied observed panels and audited 1500-step prefixes are a development replay, never a new independent acceptance." })
    puts "Copied prepared v08 replay; import prefixes, run probes/baselines, then registered training phases."
  end

  def audit_training_isolation(root)
    verify(root)
    forbidden = JSON.parse(File.read(File.join(root, "forbidden.json"))).to_set
    Dir[File.join(root, "arms/*/result.json")].each do |file|
      result = sealed(file)
      raise "Isolation audit needs a complete endpoint" unless result.fetch("step") == 6000
      checkpoint = result.fetch("checkpoint")
      raise "Isolation audit weights mismatch" unless sha(File.join(checkpoint, "weights.pt")) == result.fetch("weights_sha256")
      state = JSON.parse(File.read(File.join(checkpoint, "metadata.json"))).fetch("training")
      hashes = state.fetch("rendering_coverage").fetch("surface_hashes").keys.to_set
      overlap = hashes & forbidden
      raise "Trained rendering overlaps a reserved world" unless overlap.empty?
      path = File.join(root, "isolation-correction", "trained-#{File.basename(File.dirname(file))}.json")
      value = { "weights_sha256" => result.fetch("weights_sha256"), "metadata_sha256" => sha(File.join(checkpoint, "metadata.json")),
        "reserved_sha256" => sha(File.join(root, "forbidden.json")), "unique_rendered_states" => hashes.size, "overlap" => overlap.size }
      File.exist?(path) ? (raise "Training-isolation audit changed" unless sealed(path) == value) : seal(path, value)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/event-generalization-v08-corrected", variant: "reference", seed: 1337, split: "development" }
  OptionParser.new do |parser|
    %i[phase output variant identity split source].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options.fetch(:output))
  begin
    case options.fetch(:phase)
    when "replay" then DecisionEventGeneralization.replay(root, File.expand_path(options.fetch(:source)))
    when "select", "finalize_choices"
      DecisionEventGeneralization.audit_training_isolation(root)
      DecisionEventGeneralization.public_send(options.fetch(:phase), root)
    when "prepare", "import_prefixes", "lockbox", "report", "plot", "crossed_templates", "audit_inputs" then DecisionEventGeneralization.public_send(options.fetch(:phase), root)
    when "preflight" then DecisionEventGeneralization.preflight(root, options.fetch(:variant))
    when "baseline" then DecisionEventGeneralization.baseline(root, options.fetch(:seed))
    when "train" then DecisionEventGeneralization.train(root, options.fetch(:variant), options.fetch(:seed))
    when "audit" then DecisionEventGeneralization.audit(root, options.fetch(:identity), split: options.fetch(:split))
    else raise ArgumentError, "Unknown v08 phase"
    end
  rescue StandardError => error
    directory = File.join(root, "failures")
    FileUtils.mkdir_p(directory)
    DecisionEventGeneralization.write(File.join(directory, "#{Time.now.utc.strftime('%Y%m%dT%H%M%S')}-#{SecureRandom.hex(4)}.json"),
      { "at" => Time.now.utc.iso8601, "options" => options, "class" => error.class.name, "message" => error.message, "backtrace" => error.backtrace.first(10) })
    raise
  end
end
