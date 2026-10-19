module DecisionRelational
  module_function

  def report(root)
    verify(root)
    results = Dir.glob(File.join(root, "fit-*/result.json")).sort.to_h do |path|
      name = File.basename(File.dirname(path)).delete_prefix("fit-")
      result = JSON.parse(File.read(path))
      transfer = JSON.parse(File.read(File.join(File.dirname(path), "transfer.json")))
      [name, result.merge("transfer" => transfer, "result_sha256" => sha(path))]
    end
    ledger = { "fitting_updates" => results.values.sum { |result| result.fetch("step") }, "fits" => results,
      "acceptance_opened" => Dir.glob(File.join(root, "**/acceptance-opened.json")).any?,
      "scope" => "Result entries describe completed fits; trace totals also count incomplete and replayed updates. Fitting and observed development do not establish independent generalization." }
    ledger["successful_updates_by_trace"] = update_ledger(root)
    ledger["total_successful_updates"] = ledger.fetch("successful_updates_by_trace").values.sum
    ledger["update_ceiling"] = UPDATE_LIMIT
    write(File.join(root, "ledger.json"), ledger)
    write(File.join(root, "compact-results.json"), compact_results(root, ledger))
    puts JSON.pretty_generate(ledger.except("fits"))
    results.each do |name, result|
      last = result.fetch("measurements").last
      puts "#{name}: steps=#{result.fetch('step')} passed=#{result.fetch('passed')} fit=#{last.fetch('by_language').transform_values { |cell| cell.slice('balanced', 'actor_binding_all_correct', 'event_binding_all_correct') }}"
    end
    ledger
  end

  def compact_results(root, ledger)
    protocol = JSON.parse(File.read(File.join(root, "protocol.json")))
    fields = %w[count balanced accuracy actor_binding_all_correct event_binding_all_correct recall]
    compact = ->(metrics) { metrics.fetch("by_language").transform_values { |cell| cell.slice(*fields) } }
    fits = ledger.fetch("fits").transform_values do |result|
      { "updates" => result.fetch("step"), "passed" => result.fetch("passed"), "result_sha256" => result.fetch("result_sha256"),
        "training" => compact.call(result.fetch("measurements").last), "observed_development" => compact.call(result.fetch("observed_development")) }
    end
    pilots = Dir.glob(File.join(root, "pilot*/seed-*/summary.json")).sort.to_h do |path|
      summary = JSON.parse(File.read(path))
      history = JSON.parse(File.read(File.join(File.dirname(path), "development.json"))).last
      profiles = history.fetch("profiles").transform_values { |value| value.transform_values { |metrics| compact.call(metrics) } }
      [File.dirname(path).delete_prefix(root + "/"), summary.slice("step", "stage", "known_mastery_passed", "stop_reason", "device", "known_updates", "unknown_updates", "examples_seen").merge(
        "summary_sha256" => sha(path), "eligible_profiles" => summary.fetch("selected").keys,
        "pilot_protocol_sha256" => sha(File.join(File.dirname(File.dirname(path)), "protocol.json")),
        "actual_trace_updates" => ledger.fetch("successful_updates_by_trace").fetch(File.join(File.dirname(path).delete_prefix(root + "/"), "choice/training.jsonl")),
        "final_training_probe" => compact.call(history.fetch("training_probe")), "final_observed_development" => profiles)]
    end
    { "protocol_sha256" => sha(File.join(root, "protocol.json")),
      "parent_weights_sha256" => protocol.fetch("parent_weights_sha256"), "tokenizer_fingerprint" => protocol.fetch("tokenizer_fingerprint"),
      "total_successful_updates" => ledger.fetch("total_successful_updates"), "update_ceiling" => UPDATE_LIMIT,
      "acceptance_opened" => ledger.fetch("acceptance_opened"), "fits" => fits, "completed_pilots" => pilots,
      "runtime_amendments_sha256" => Dir.glob(File.join(root, "pilot*/evaluation-runtime-amendment.json")).sort.to_h { |path| [path.delete_prefix(root + "/"), sha(path)] },
      "scope" => "Compact extract from frozen local results; development is observed, not independent acceptance. Interrupted or incomplete pilots are excluded. Raw data, configs, sources and traces remain in the local experiment root." }
  end
end
