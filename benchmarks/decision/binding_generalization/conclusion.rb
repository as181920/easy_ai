module DecisionBindingGeneralization
  module_function

  def conclude(root)
    protocol = verify(root)
    archive_integrity = verify_source_archives(root, protocol)
    selection = decision(root)
    raise "Negative primary result requires the separately frozen pretrained diagnostic" unless selection.fetch("improving")
    selected = selection.fetch("winner")
    records = publish_report(root)
    confirmations = [1337, 2027].to_h do |seed|
      reference = records.fetch("results").fetch("arms/ce-#{seed}/result-6000.json")
      candidate = records.fetch("results").fetch("arms/#{selected}-#{seed}/result-6000.json")
      raise "Confirmation row sampling differs" unless reference.fetch("row_visits_sha256") == candidate.fetch("row_visits_sha256")
      [seed, direction?(candidate, reference, protocol.fetch("direction"))]
    end
    endpoints = records.fetch("results").to_h do |name, result|
      identity = "#{result.fetch('variant')}-#{result.fetch('seed')}-#{result.fetch('step')}"
      checks = %w[cpu cuda].to_h do |device|
        file = File.join(root, "audit", identity, "#{device}.json")
        audit = JSON.parse(File.read(file))
        raise "Audit/result changed" unless audit.fetch("result_sha256") == result.fetch("result_sha256")
        raise "Audit/weights changed" unless audit.fetch("weights_sha256") == sha(File.join(result.fetch("checkpoint"), "weights.pt"))
        raise "Recorded classification differs" unless audit.fetch("views").values.all? { |view| view.fetch("matches_recorded_classification_metrics") }
        raise "Public/batched prediction differs" unless audit.fetch("public_interface").all? { |cell| cell.fetch("batch_public_prediction_agrees") }
        [device, audit.merge("audit_sha256" => sha(file))]
      end
      compact = result.slice("variant", "seed", "step", "device", "parameters", "known_mastery", "mean_original_row_visits", "unvisited_original_rows", "row_visits_sha256", "input_tokens", "result_sha256")
      compact["weights_sha256"] = checks.fetch("cpu").fetch("weights_sha256")
      compact["training_full"] = DecisionScaling.compact_metrics(result.fetch("training_full"))
      compact["development"] = result.fetch("development").transform_values { |metric| DecisionScaling.compact_metrics(metric) }
      compact["audits"] = checks
      compact["cross_device"] = compare_audit_devices(File.join(root, "audit", identity))
      [name, compact]
    end
    exposure = [1337, 2027].to_h do |seed|
      path = File.join(root, "audit", "ce-#{seed}-1500", "cpu.json")
      early = JSON.parse(File.read(path))
      final = records.fetch("results").fetch("arms/ce-#{seed}/result-6000.json")
      [seed, { "early_weights_sha256" => early.fetch("weights_sha256"), "early_audit_sha256" => sha(path),
        "step_1500" => early.fetch("development").transform_values { |metric| DecisionScaling.compact_metrics(metric) },
        "step_6000" => final.fetch("development").transform_values { |metric| DecisionScaling.compact_metrics(metric) } }]
    end
    value = { "status" => "completed", "selected_variant" => selected, "direction_by_seed" => confirmations,
      "direction_repeated" => confirmations.values.all?, "selection" => selection, "endpoints" => endpoints,
      "within_data_ce_learning_curve" => exposure, "protocol_sha256" => records.fetch("protocol_sha256"),
      "novel_actor_error_transitions" => [1337, 2027].to_h { |seed| [seed, actor_error_transitions(root, selected, seed)] },
      "source_integrity" => records.fetch("source_integrity"), "update_ledger" => records.fetch("update_ledger"),
      "archived_source_integrity" => archive_integrity,
      "actual_optimizer_updates" => records.fetch("actual_optimizer_updates"), "update_ceiling" => UPDATE_CEILING,
      "trace_diagnostics" => records.fetch("executed_trace_diagnostics"),
      "resource_observations" => Dir[File.join(root, "arms/*/development.json")].sort.to_h do |file|
        history = JSON.parse(File.read(file))
        values = history.map { |row| row.fetch("sampled_process_mib") }
        [File.basename(File.dirname(file)), { "sampled_process_mib_min" => values.min, "sampled_process_mib_max" => values.max,
          "process_soft_budget_mib" => 4096, "scope" => "Per-process observations at fixed evaluation boundaries, not continuous peak memory or exclusive GPU throughput." }]
      end,
      "amendments" => records.select { |key, _| key.end_with?("/amendment", "/replay-audit") },
      "pretrained_diagnostic" => "Not triggered: the complete primary factorial produced an improving own-model recipe. No pretrained weights were downloaded or used.",
      "acceptance_opened" => false, "release_changed" => false,
      "scope" => "Observed development, two training seeds, one fixed dataset/panel. Pair counts share worlds. No fresh acceptance, calibrated factual confidence, natural-language coverage or Jev parity claim." }
    write(File.join(root, "conclusion.json"), value)
    write("docs/decision/v07-results.json", value)
    puts "Completed v07 evidence: repeated direction=#{value.fetch('direction_repeated')}, #{value.fetch('actual_optimizer_updates')} actual updates"
    value
  end

  def verify_source_archives(root, protocol)
    manifests = { "source-frozen" => protocol.fetch("source_sha256") }
    { "source-runtime-frozen" => "namespace-correction", "source-reference-policy-frozen" => "exposure-confirmation", "selection-integrity/source-after" => "selection-integrity", "reporting-integrity/source-after" => "reporting-integrity" }.each do |directory, amendment|
      path = File.join(root, amendment, "amendment.json")
      manifests[directory] = JSON.parse(File.read(path)).fetch("runtime_source_sha256") if File.exist?(path)
    end
    manifests.to_h do |directory, hashes|
      hashes.each { |file, digest| raise "Archived source changed: #{directory}/#{file}" unless sha(File.join(root, directory, file)) == digest }
      [directory, { "verified_files" => hashes.size }]
    end
  end

  def compare_audit_devices(directory)
    %w[canonical seen_combinations novel_actor].to_h do |name|
      cpu = DecisionRelational.rows(File.join(directory, "cpu-#{name}.jsonl"))
      cuda = DecisionRelational.rows(File.join(directory, "cuda-#{name}.jsonl"))
      raise "Audit device row mismatch" unless cpu.map { |row| row.fetch("id") } == cuda.map { |row| row.fetch("id") }
      delta = cpu.zip(cuda).flat_map do |left, right|
        raise "Audit device options mismatch" unless left.fetch("options") == right.fetch("options")
        left.fetch("logits").zip(right.fetch("logits")).map { |a, b| (a - b).abs }
      end.max
      [name, { "rows" => cpu.size, "prediction_differences" => cpu.zip(cuda).count { |a, b| a.fetch("prediction") != b.fetch("prediction") }, "max_absolute_logit_delta" => delta }]
    end
  end

  def actor_error_transitions(root, selected, seed)
    reference = DecisionRelational.rows(File.join(root, "audit", "ce-#{seed}-6000", "cpu-novel_actor.jsonl"))
    candidate = DecisionRelational.rows(File.join(root, "audit", "#{selected}-#{seed}-6000", "cpu-novel_actor.jsonl")).to_h { |row| [row.fetch("id"), row] }
    reference.select { |row| row.fetch("source").end_with?("Actor") && row.dig("world", "mixed") }.group_by { |row| row.fetch("language") }.transform_values do |rows|
      counts = { "both_correct" => 0, "corrected_pairs" => 0, "lost_pairs" => 0, "neither_correct" => 0 }
      samples = []
      rows.group_by { |row| [row.fetch("state"), row.dig("world", "assertion")] }.each_value do |pair|
        raise "Transition pair invalid" unless pair.size == 2 && pair.map { |row| row.fetch("target") }.sort == %w[no yes]
        before = pair.all? { |row| row.fetch("prediction") == row.fetch("target") }
        after = pair.all? { |row| candidate.fetch(row.fetch("id")).fetch("prediction") == row.fetch("target") }
        key = before ? (after ? "both_correct" : "lost_pairs") : (after ? "corrected_pairs" : "neither_correct")
        counts[key] += 1
        next if after || samples.size >= 2
        samples << pair.map { |row| row.slice("id", "state", "question", "target").merge("ce_prediction" => row.fetch("prediction"), "candidate_prediction" => candidate.fetch(row.fetch("id")).fetch("prediction")) }
      end
      counts.merge("remaining_failure_examples" => samples, "scope" => "Post-hoc fixed-panel paired comparison, not additional independent worlds or a new selection rule.")
    end
  end
end
