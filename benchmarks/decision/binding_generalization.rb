#!/usr/bin/env ruby
require_relative "scaling"
require_relative "binding_generalization/trainer"
require_relative "binding_generalization/protocol"
require_relative "binding_generalization/experiment"
require_relative "binding_generalization/presentation"
require_relative "binding_generalization/audit"
require_relative "binding_generalization/conclusion"

module DecisionBindingGeneralization
  ROOT_V06 = File.expand_path("runs/decision/scaling-v06")
  VARIANTS = %w[ce rename pair combined].freeze
  UPDATE_CEILING = 42_500
  module_function

  def write(path, value)
    DecisionScaling.write(path, value)
  end

  def sha(path)
    DecisionScaling.sha(path)
  end

  def dispatch_train(root, variant, seed, steps)
    variant = variant.to_s
    raise ArgumentError, "Unregistered arm" unless VARIANTS.include?(variant) && [1337, 2027].include?(seed) && [6000, 12000].include?(steps)
    FileUtils.mkdir_p(File.join(root, "locks"))
    File.open(File.join(root, "locks", "#{variant}-#{seed}.lock"), "w") do |lock|
      lock.flock(File::LOCK_EX)
      path = File.join(root, "arms", "#{variant}-#{seed}", "result-#{steps}.json")
      if File.exist?(path)
        verify(root)
        result = JSON.parse(File.read(path))
        raise "Completed arm identity mismatch" unless result.values_at("variant", "seed", "step") == [variant, seed, steps]
        puts "Preserved completed #{variant}/#{seed}/#{steps}; no repeated training"
      else
        train(root, variant, seed, steps)
      end
    end
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "prepare", output: "runs/decision/generalization-v07", variant: "ce", seed: 1337, steps: 6000 }
  OptionParser.new do |parser|
    %i[phase output variant device].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    %i[seed steps].each { |key| parser.on("--#{key} N", Integer) { |value| options[key] = value } }
  end.parse!
  root = File.expand_path(options.fetch(:output))
  begin
    case options.fetch(:phase)
    when "prepare" then DecisionBindingGeneralization.prepare(root)
    when "preflight" then DecisionBindingGeneralization.preflight(root)
    when "train" then DecisionBindingGeneralization.dispatch_train(root, options.fetch(:variant), options.fetch(:seed), options.fetch(:steps))
    when "select" then DecisionBindingGeneralization.select(root)
    when "report" then DecisionBindingGeneralization.publish_report(root)
    when "plot" then DecisionBindingGeneralization.plot(root)
    when "conclude" then DecisionBindingGeneralization.conclude(root)
    when "audit" then DecisionBindingGeneralization.audit_endpoint(root, options.fetch(:variant), options.fetch(:seed), options.fetch(:steps), device: options.fetch(:device, "cpu"))
    when "audit-early" then DecisionBindingGeneralization.audit_early(root, options.fetch(:variant), options.fetch(:seed), options.fetch(:steps), device: options.fetch(:device, "cpu"))
    else raise ArgumentError, "Unknown generalization phase"
    end
  rescue StandardError => error
    directory = File.join(root, "failures")
    FileUtils.mkdir_p(directory)
    DecisionBindingGeneralization.write(File.join(directory, "#{Time.now.utc.strftime('%Y%m%dT%H%M%S')}-#{SecureRandom.hex(4)}.json"),
      { "options" => options, "class" => error.class.name, "message" => error.message, "backtrace" => error.backtrace.first(8) })
    raise
  end
end
