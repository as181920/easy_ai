#!/usr/bin/env ruby
require_relative "event_generalization"
require_relative "expression_retention/catalog"
require_relative "expression_retention/corpus"
require_relative "expression_retention/trainer"
require_relative "expression_retention/protocol"
require_relative "expression_retention/experiment"
require_relative "expression_retention/presentation"

module DecisionExpressionRetention
  VARIANTS = %w[reference events expressions both no_replay].freeze
  STEPS = 6000
  CEILING = 43_000
  ROOT_V08 = File.expand_path("runs/decision/event-generalization-v08-corrected")
  module_function

  def write(path, value)
    DecisionEventGeneralization.write(path, value)
  end

  def sha(path)
    DecisionEventGeneralization.sha(path)
  end

  def seal(path, value)
    DecisionEventGeneralization.seal(path, value)
  end

  def sealed(path)
    DecisionEventGeneralization.sealed(path)
  end

  def raw(root, name)
    DecisionEventGeneralization.raw(root, name)
  end

  def parent(seed)
    DecisionEventGeneralization.parent(seed)
  end

  def independent_pairs(rows, guesses, axis:)
    DecisionEventGeneralization.independent_pairs(rows, guesses, axis: axis)
  end

  def metrics(rows, logits)
    value = DecisionJudgment.metrics(rows, logits)
    value["by_form"] = rows.each_index.group_by { |index| rows[index].dig("world", "form") || 0 }.transform_values do |indices|
      DecisionJudgment.metrics(indices.map { |index| rows[index] }, indices.map { |index| logits[index] })
    end.transform_keys(&:to_s)
    checks = { "all" => rows.each_index.to_a }.merge(rows.each_index.group_by { |index| (rows[index].dig("world", "form") || 0).to_s })
    checks.each do |form, indices|
      subset = indices.map { |index| rows[index] }
      guesses = indices.map { |index| rows[index].fetch("options")[logits[index].each_index.max_by { |option| logits[index][option] }].fetch("id") }
      panel = form == "all" ? value : value.fetch("by_form").fetch(form)
      %w[actor event].each do |axis|
        independent_pairs(subset, guesses, axis: axis).each do |language, cell|
          raise "Independent per-template pairs mismatch" unless cell.fetch("groups") == panel.dig("by_language", language, "#{axis}_binding_groups") && cell.fetch("all_correct") == panel.dig("by_language", language, "#{axis}_binding_all_correct")
        end
      end
    end
    value
  end

  def target_cells(value)
    %w[events expressions].flat_map do |view|
      value.fetch("development").fetch(view).fetch("by_form").sort.flat_map do |form, panel|
        Catalog::LANGUAGES.flat_map do |language|
          %w[actor event].map { |axis| ["#{view}/#{form}/#{language}/#{axis}", panel.dig("by_language", language, "#{axis}_binding_all_correct")] }
        end
      end
    end.to_h
  end

  def direction?(candidate, reference)
    a, b = target_cells(candidate), target_cells(reference)
    raise "Incompatible direction panels" unless a.keys == b.keys && (a.values + b.values).all? { |value| value.is_a?(Numeric) }
    return false unless a.values.min + 1e-12 >= b.values.min + 0.05
    return false unless a.all? { |key, value| value + 0.05 + 1e-12 >= b.fetch(key) }
    panels = reference.fetch("development").except("composed").transform_values { |panel| panel.fetch("by_form") }
    return false unless panels.all? do |view, forms|
      forms.all? do |form, panel|
        Catalog::LANGUAGES.all? do |language|
          candidate.dig("development", view, "by_form", form, "by_language", language, "balanced") + 0.03 + 1e-12 >= panel.dig("by_language", language, "balanced")
        end
      end
    end
    reference.fetch("regression").all? do |view, panel|
      Catalog::LANGUAGES.all? do |language|
        before = panel.fetch("by_language").fetch(language)
        after = candidate.dig("regression", view, "by_language", language)
        after.fetch("balanced") + 0.03 + 1e-12 >= before.fetch("balanced") &&
          %w[actor event].all? { |axis| after.fetch("#{axis}_binding_all_correct") + 0.05 + 1e-12 >= before.fetch("#{axis}_binding_all_correct") }
      end
    end
  end

  def rank(value)
    cells = target_cells(value).values
    [cells.min, cells.sum.fdiv(cells.size), value.fetch("development").except("composed").values.map { |panel| panel.fetch("worst_language_balanced") }.min,
      -value.fetch("development").except("composed").values.sum { |panel| panel.fetch("nll") }]
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/expression-retention-v09", variant: "reference", seed: 1337, split: "development" }
  OptionParser.new do |parser|
    %i[phase output variant identity split source].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options.fetch(:output))
  begin
    case options.fetch(:phase)
    when "prepare", "select", "finalize_choices", "lockbox", "report", "plot" then DecisionExpressionRetention.public_send(options.fetch(:phase), root)
    when "baseline" then DecisionExpressionRetention.baseline(root, options.fetch(:seed))
    when "preflight" then DecisionExpressionRetention.preflight(root, options.fetch(:variant))
    when "train" then DecisionExpressionRetention.train(root, options.fetch(:variant), options.fetch(:seed))
    when "audit" then DecisionExpressionRetention.audit(root, options.fetch(:identity), split: options.fetch(:split))
    when "replay" then DecisionExpressionRetention.replay(root, File.expand_path(options.fetch(:source)))
    else raise ArgumentError, "Unknown v09 phase"
    end
  rescue StandardError => error
    DecisionExpressionRetention.write(File.join(root, "failures", "#{Time.now.utc.strftime('%Y%m%dT%H%M%S')}-#{SecureRandom.hex(4)}.json"),
      { "at" => Time.now.utc.iso8601, "options" => options, "class" => error.class.name, "message" => error.message, "backtrace" => error.backtrace.first(10) })
    raise
  end
end
