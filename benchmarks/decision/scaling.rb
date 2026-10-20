#!/usr/bin/env ruby
require_relative "relational_encoding"
require_relative "scaling/initialization"
require_relative "scaling/memory"
require_relative "scaling/data"
require_relative "scaling/quad_trainer"
require_relative "scaling/preparation"
require_relative "scaling/budget"
require_relative "scaling/training"
require_relative "scaling/selection"
require_relative "scaling/report"
require_relative "scaling/plot"
require_relative "scaling/natural_preparation"
require_relative "scaling/supervised_trainer"
require_relative "scaling/supervised_preparation"
require_relative "scaling/supervised"
require_relative "scaling/supervised_selection"
require_relative "scaling/parameter_audit"
require_relative "scaling/runtime"
require_relative "scaling/supervised_plot"
require_relative "scaling/padding_audit"
require_relative "scaling/retained_trainer"
require_relative "scaling/retention"
require_relative "scaling/diagnostic_policy"
require_relative "scaling/binding_audit"

module DecisionScaling
  PARENT = DecisionRelational::PARENT
  PROFILES = { "base" => [4, 768], "medium" => [8, 1536], "large" => [12, 2048] }.freeze
  UPDATE_LIMIT = 30_000
  module_function

  def write(path, value)
    DecisionRelational.write(path, value)
  end

  def sha(path)
    DecisionRelational.sha(path)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare-memory", output: "runs/decision/scaling-v06", profile: "base", microbatch: 4, seed: 1337, scale: 32, limit: 1500 }
  OptionParser.new do |parser|
    %i[phase output profile split device].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    %i[microbatch seed scale limit].each { |key| parser.on("--#{key} N", Integer) { |value| options[key] = value } }
  end.parse!
  root = File.expand_path(options.fetch(:output))
  case options.fetch(:phase)
  when "prepare-memory" then DecisionScaling.prepare_memory(root)
  when "prepare" then DecisionScaling.prepare_development(root)
  when "baseline" then DecisionScaling.baseline(root)
  when "extensions" then puts JSON.pretty_generate(DecisionScaling.select_extensions(root))
  when "select" then puts JSON.pretty_generate(DecisionScaling.final_controlled_selection(root))
  when "report" then DecisionScaling.scaling_report(root)
  when "plot" then DecisionScaling.scaling_plot(root)
  when "prepare-natural-review" then DecisionScaling.prepare_natural_review(root)
  when "order-natural-review" then DecisionScaling.order_natural_review(root)
  when "import-natural-review" then DecisionScaling.import_natural_review(root, options.fetch(:split))
  when "prepare-supervised" then DecisionScaling.prepare_supervised(root)
  when "supervised-baseline" then DecisionScaling.supervised_baseline(root)
  when "supervised" then DecisionScaling.supervised(root, options.fetch(:profile), seed: options.fetch(:seed))
  when "select-supervised" then puts JSON.pretty_generate(DecisionScaling.select_supervised(root))
  when "audit-parameters" then DecisionScaling.parameter_audit(root)
  when "diagnostic-runtime" then DecisionScaling.diagnostic_runtime(root, device: options.fetch(:device, "cpu"))
  when "plot-supervised" then DecisionScaling.supervised_plot(root)
  when "audit-padding" then DecisionScaling.padding_audit(root)
  when "prepare-retained" then DecisionScaling.prepare_retained(root)
  when "retained" then DecisionScaling.retained(root)
  when "select-retained" then puts JSON.pretty_generate(DecisionScaling.select_retained(root))
  when "audit-diagnostic-policy" then puts JSON.pretty_generate(DecisionScaling.diagnostic_policy_audit(root))
  when "audit-binding" then DecisionScaling.binding_audit(root, options.fetch(:profile))
  when "audit-binding-inputs" then DecisionScaling.binding_input_audit(root)
  when "train" then DecisionScaling.train(root, options.fetch(:profile), options.fetch(:scale), seed: options.fetch(:seed), limit: options.fetch(:limit))
  when "prepare-data" then DecisionScaling.prepare_data(root)
  when "prepare-seen" then DecisionScaling.prepare_seen(root)
  when "audit-limits" then DecisionScaling.audit_limits(root)
  when "audit-cross-round" then DecisionScaling.cross_round_audit(root)
  when "memory" then DecisionScaling.memory(root, options.fetch(:profile), options.fetch(:microbatch))
  when "memory-controlled" then DecisionScaling.memory(root, options.fetch(:profile), options.fetch(:microbatch), controlled: true)
  else raise ArgumentError, "Unknown scaling phase"
  end
end
