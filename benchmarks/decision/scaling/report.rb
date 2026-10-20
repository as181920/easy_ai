module DecisionScaling
  module_function

  def compact_metrics(value)
    fields = %w[count accuracy balanced recall actor_binding_groups actor_binding_all_correct event_binding_groups event_binding_all_correct]
    value.fetch("by_language").transform_values { |cell| cell.slice(*fields) }
  end

  def trace_audit(root)
    update_ledger(root).to_h do |relative, count|
      rows = File.foreach(File.join(root, relative)).map { |line| JSON.parse(line) }
      finite = rows.all? { |row| row.values.grep(Numeric).all?(&:finite?) }
      cuda = rows.all? { |row| row.fetch("device") == "cuda" }
      raise "Invalid optimizer trace: #{relative}" unless finite && cuda
      [relative, { "actual_updates" => count, "finite_numeric_fields" => finite, "all_cuda" => cuda, "sha256" => sha(File.join(root, relative)) }]
    end
  end

  def controlled_exposure(root, cell, result)
    scale = Integer(cell.split("-").fetch(1))
    rows = DecisionRelational.rows(File.join(root, "data", "train-#{scale}-binary.jsonl"))
    metadata = JSON.parse(File.read(File.join(result.fetch("checkpoint"), "metadata.json")))
    coverage = metadata.fetch("training").fetch("coverage")
    visits = coverage.fetch("row_visits")
    unless visits.size == rows.size && visits.all? { |count| count.is_a?(Integer) && count >= 0 } && visits.sum == result.fetch("examples_seen") && coverage.fetch("input_tokens") == result.fetch("input_tokens_seen")
      raise "Controlled exposure accounting mismatch"
    end
    buckets = rows.each_index.group_by { |index| [rows[index].fetch("language"), rows[index].fetch("source").split("-").last].join("/") }
    { "rows" => rows.size, "visited_rows" => visits.count(&:positive?), "unvisited_rows" => visits.count(&:zero?),
      "minimum_visits" => visits.min, "maximum_visits" => visits.max, "mean_visits" => visits.sum.fdiv(visits.size),
      "visits_by_language_axis" => buckets.transform_values { |indices| indices.sum { |index| visits[index] } },
      "metadata_sha256" => sha(File.join(result.fetch("checkpoint"), "metadata.json")),
      "scope" => "Sampling with replacement; mean visits is not a completed-epoch count. Training probe covers only sixteen worlds per language." }
  end

  def factorial_exposure_match(root, protocol)
    paths = protocol.fetch("profiles").product(protocol.fetch("data_scales")).to_h do |profile, scale|
      [[profile, scale], File.join(root, "arms", "#{profile}-#{scale}-1337/screen.json")]
    end
    return { "complete" => false } unless paths.values.all? { |path| File.exist?(path) }
    comparisons = protocol.fetch("data_scales").to_h do |scale|
      cells = protocol.fetch("profiles").map do |profile|
        value = JSON.parse(File.read(paths.fetch([profile, scale])))
        state = JSON.parse(File.read(File.join(value.fetch("checkpoint"), "metadata.json"))).fetch("training")
        [state.fetch("step"), state.fetch("examples_seen"), state.fetch("coverage")]
      end
      raise "Factorial capacities did not receive identical exposure" unless cells.uniq.size == 1
      [scale.to_s, { "identical_row_visits_and_input_tokens" => true, "updates" => cells.first[0], "examples_seen" => cells.first[1] }]
    end
    { "complete" => true, "by_data_scale" => comparisons, "scope" => "Initial endpoints only; identical sample counts and total unpadded input tokens, not identical model FLOPs or elapsed time." }
  end

  def scaling_report(root)
    protocol = verify(root)
    changed = protocol.fetch("source_sha256").filter_map do |file, digest|
      [file, { "frozen_sha256" => digest, "current_sha256" => sha(file) }] unless sha(file) == digest
    end.to_h
    raise "Frozen learning implementation changed" unless (changed.keys - ["benchmarks/decision/scaling.rb"]).empty?
    traces = trace_audit(root)
    results = Dir[File.join(root, "arms/*/{screen,extended}.json")].sort.to_h do |path|
      value = JSON.parse(File.read(path))
      name = File.dirname(path).delete_prefix(File.join(root, "arms") + "/") + "/" + File.basename(path, ".json")
      history = JSON.parse(File.read(File.join(File.dirname(path), "development.json"))).select { |cell| cell.fetch("step") <= value.fetch("step") }
      fields = %w[step parameters known_mastery_passed controlled_eligible device examples_seen input_tokens_seen scope]
      cell = File.basename(File.dirname(path))
      [name, value.slice(*fields).merge("sha256" => sha(path), "exposure" => controlled_exposure(root, cell, value), "development" => value.fetch("development").transform_values { |metrics| compact_metrics(metrics) },
        "final_training_probe" => compact_metrics(value.fetch("final_training_probe")),
        "measured_history" => history.map { |cell| cell.slice("step", "loss", "mastered", "stable_mastery_checks", "sampled_process_mib", "session_seconds") })]
    end
    memory = protocol.dig("hardware", "results_sha256").keys.to_h do |relative|
      value = JSON.parse(File.read(File.join(root, relative)))
      measurements = value.fetch("measurements", [])
      [File.dirname(relative).delete_prefix("preflight/"), value.slice("passed", "parameters", "microbatch", "error", "completed_updates", "scope").merge(
        "sampled_process_mib_max" => measurements.flat_map { |cell| cell.fetch("sampled_process_mib") }.compact.max,
        "successful_trace_updates" => traces.fetch(File.join(File.dirname(relative), "training.jsonl"), {}).fetch("actual_updates", 0))]
    end
    value = { "protocol_sha256" => sha(File.join(root, "protocol.json")), "parent_sha256" => protocol.fetch("parent_sha256"),
      "tokenizer_fingerprint" => protocol.fetch("tokenizer_fingerprint"), "data_audit_sha256" => protocol.fetch("data_audit_sha256"),
      "cross_round_audit_sha256" => protocol.fetch("cross_round_audit_sha256"), "plan_sha256" => protocol.fetch("plan_sha256"),
      "hardware" => protocol.fetch("hardware"), "memory" => memory, "controlled_results" => results,
      "factorial_exposure_match" => factorial_exposure_match(root, protocol),
      "initial_source_audit" => { "unchanged_files" => protocol.fetch("source_sha256").size - changed.size, "changed_entrypoints" => changed,
        "scope" => "The initial library/model/data/optimizer/sampler/training implementation remains byte-identical. The CLI entrypoint adds reporting and separately frozen conditional-phase dispatch; its original snapshot remains archived." },
      "trace_audit" => traces, "total_successful_updates" => traces.values.sum { |cell| cell.fetch("actual_updates") }, "update_ceiling" => UPDATE_LIMIT,
      "acceptance_opened" => Dir[File.join(root, "**/acceptance-opened.json")].any?,
      "scope" => "Completed controlled screens and their separately registered extensions. All successful trace lines count, including probes and replay. Development is observed; controlled-only results do not establish a factual release. Memory samples are not continuous peaks." }
    %w[extension-selection selection supervised-selection retained-selection confirmation].each do |name|
      path = File.join(root, "#{name}.json")
      value[name] = JSON.parse(File.read(path)).merge("sha256" => sha(path)) if File.exist?(path)
    end
    value["supervised_results"] = Dir[File.join(root, "supervised/*/summary.json")].sort.to_h do |path|
      summary = JSON.parse(File.read(path))
      last = JSON.parse(File.read(File.join(File.dirname(path), "development.json"))).last
      profiles = last.fetch("profiles").transform_values do |profile|
        profile.merge("controlled" => profile.fetch("controlled").transform_values { |metrics| compact_metrics(metrics) },
          "natural" => compact_metrics(profile.fetch("natural")), "regression" => compact_metrics(profile.fetch("regression")))
      end
      [File.basename(File.dirname(path)), summary.except("selected", "checkpoint").merge("sha256" => sha(path), "eligible_profiles" => summary.fetch("selected").keys,
        "final_training_probe" => compact_metrics(last.fetch("training_probe")), "final_observed_development" => profiles)]
    end
    value["natural_review"] = %w[train development].filter_map do |name|
      path = File.join(root, "natural-review", "#{name}-audit.json")
      next unless File.exist?(path)
      audit = JSON.parse(File.read(path))
      [name, audit.slice("qualified_counts", "by_source", "reviewer_scope").merge("sha256" => sha(path))]
    end.to_h
    value["diagnostic_artifacts_sha256"] = %w[initialization-audit.json parameter-audit.json padding-audit.json diagnostic-runtime-cpu.json diagnostic-runtime-cuda.json supervised-protocol.json retained-protocol.json diagnostic-policy-audit.json].filter_map do |name|
      path = File.join(root, name)
      [name, sha(path)] if File.exist?(path)
    end.to_h
    inventory_note = File.join(root, "natural-review/inventory-interpretation.json")
    value["natural_inventory_interpretation"] = JSON.parse(File.read(inventory_note)).merge("sha256" => sha(inventory_note)) if File.exist?(inventory_note)
    value["binding_correctness_audit"] = Dir[File.join(root, "binding-audit/*.json")].sort.to_h do |path|
      [File.basename(path, ".json"), JSON.parse(File.read(path)).except("public_predictor_checks").merge("sha256" => sha(path))]
    end
    policy_path = File.join(root, "diagnostic-policy-audit.json")
    if File.exist?(policy_path)
      policy = JSON.parse(File.read(policy_path))
      value["development_policy_assessment"] = policy.slice("amendment", "original_rules", "development_rules", "scope").merge(
        "first_stable_steps" => policy.fetch("histories").transform_values { |item| item.slice("original_90", "development_85").transform_values { |history| history.fetch("first_stable_step") } })
    end
    raise "Update ceiling exceeded" if value.fetch("total_successful_updates") > UPDATE_LIMIT
    write(File.join(root, "compact-results.json"), value)
    puts "Scaling report: #{results.size} completed endpoints; #{value.fetch('total_successful_updates')} actual CUDA optimizer updates"
    value
  end
end
