#!/usr/bin/env ruby
require_relative "judgment"
require_relative "relational_encoding/protocol"
require_relative "relational_encoding/fitting"
require_relative "relational_encoding/audit"
require_relative "relational_encoding/followups"
require_relative "relational_encoding/selection"
require_relative "relational_encoding/report"
require_relative "relational_encoding/runtime"
require_relative "relational_encoding/plot"
require_relative "relational_encoding/pilot_protocol"
require_relative "relational_encoding/pilot"
require_relative "relational_encoding/focused_trainer"
require_relative "relational_encoding/budget"
require_relative "relational_encoding/generalization_plot"
require_relative "relational_encoding/uncertainty"
require_relative "relational_encoding/calibration"
require_relative "relational_encoding/acceptance"
require_relative "relational_encoding/token_accounting"
require_relative "relational_encoding/composition"

module DecisionRelational
  SOURCE = "runs/decision/judgment-v04/pilot"
  PARENT = DecisionJudgment::PARENT
  ARMS = %w[separate joint].freeze
  module_function

  def write(path, value)
    DecisionJudgment.write(path, value)
  end

  def rows(path)
    DecisionJudgment.raw(path)
  end

  def sha(path)
    DecisionJudgment.sha(path)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/relational-v05", arm: "separate", seed: 1337, variant: "standard", selection: "default" }
  OptionParser.new do |parser|
    %i[phase output arm variant selection].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options.fetch(:output))
  case options.fetch(:phase)
  when "prepare" then DecisionRelational.prepare(root)
  when "audit" then DecisionRelational.audit(root)
  when "report" then DecisionRelational.report(root)
  when "account-fit-tokens" then DecisionRelational.account_fit_tokens(root)
  when "plot" then DecisionRelational.plot(root)
  when "plot-generalization" then DecisionRelational.generalization_plot(root)
  when "calibrate" then DecisionRelational.calibrate(root, variant: options.fetch(:variant))
  when "accept" then DecisionRelational.accept(root, variant: options.fetch(:variant))
  when "package" then DecisionRelational.package_preview(root, variant: options.fetch(:variant))
  when "prepare-pilot" then DecisionRelational.prepare_pilot(root, variant: options.fetch(:variant))
  when "pilot" then DecisionRelational.pilot(root, options.fetch(:seed), variant: options.fetch(:variant))
  when "select-pilot" then puts JSON.pretty_generate(DecisionRelational.select_pilot(root, variant: options.fetch(:variant)))
  when "select-fit" then puts JSON.pretty_generate(DecisionRelational.select_fit(root, selection: options.fetch(:selection)))
  when "freeze-recipe" then DecisionRelational.freeze_recipe(root, options.fetch(:arm))
  when "fit" then DecisionRelational.fit(root, options.fetch(:arm), options.fetch(:seed), selection: options.fetch(:selection))
  when "runtime" then DecisionRelational.runtime(root, options.fetch(:arm), options.fetch(:seed))
  when "composition" then DecisionRelational.composition(root, options.fetch(:arm), options.fetch(:seed))
  else raise ArgumentError, "Unknown relational phase"
  end
end
