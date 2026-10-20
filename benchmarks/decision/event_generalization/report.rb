module DecisionEventGeneralization
  module_function

  def report(root)
    protocol = verify(root)
    raise "Frozen plan changed" unless sha(File.join(root, "plan-frozen.md")) == protocol.fetch("plan_sha256")
    choices = sealed(File.join(root, "final-choices.json"))
    decision = selection(root)
    traces = ledger(root)
    diagnostics = traces.to_h do |relative, count|
      values = File.foreach(File.join(root, relative)).map { |line| JSON.parse(line) }
      raise "Invalid experiment CUDA trace" unless values.all? { |row| row.fetch("device") == "cuda" && row.values.grep(Numeric).all?(&:finite?) }
      [relative, { "updates" => count, "sha256" => sha(File.join(root, relative)), "mean_ce" => values.sum { |row| row.fetch("choice_loss") }.fdiv(count),
        "mean_actor_loss" => values.sum { |row| row.fetch("actor_pair_loss") }.fdiv(count), "mean_event_loss" => values.sum { |row| row.fetch("event_pair_loss") }.fdiv(count),
        "mean_gradient_norm" => values.sum { |row| row.fetch("gradient_norm_preclip") }.fdiv(count) }]
    end
    budget!(root)
    endpoints = Dir[File.join(root, "arms/*/result.json")].sort.to_h do |file|
      value = sealed(file)
      raise "Incomplete training endpoint" unless value.fetch("step") == 6000 && value.fetch("device") == "cuda"
      metadata = JSON.parse(File.read(File.join(value.fetch("checkpoint"), "metadata.json"))).fetch("training")
      forbidden = JSON.parse(File.read(File.join(root, "forbidden.json"))).to_set
      overlap = metadata.fetch("rendering_coverage").fetch("surface_hashes").keys.to_set & forbidden
      raise "Executed training overlaps a reserved rendering" unless overlap.empty?
      rendering = value.fetch("rendering_coverage")
      raise "Rendering exposure mismatch" unless rendering.fetch("event_rows").values.sum == value.fetch("examples_seen") && rendering.fetch("form_rows").values.sum == value.fetch("examples_seen")
      forbidden_events = Catalog::POOLS.fetch("development") + Catalog::POOLS.fetch("lockbox")
      raise "Held event appeared in training" unless rendering.fetch("event_rows").keys.none? { |key| forbidden_events.include?(key.split("/").last) }
      raise "Held expression appeared in training" unless rendering.fetch("form_rows").keys.all? { |key| Catalog::FORMS.fetch("train").include?(key.split("/").last.to_i) }
      identity = File.basename(File.dirname(file))
      audit = validated_audit(root, identity, "development")
      raise "Development audit weights mismatch" unless audit.fetch("weights_sha256") == value.fetch("weights_sha256")
      [identity, value.except("checkpoint").merge("training_reserved_rendering_overlap" => overlap.size, "development" => value.fetch("development").transform_values { |metrics| DecisionScaling.compact_metrics(metrics) },
        "regression" => value.fetch("regression").transform_values { |metrics| DecisionScaling.compact_metrics(metrics) },
        "training_full" => DecisionScaling.compact_metrics(value.fetch("training_full")), "retention_diagnostics" => retention_diagnostics(File.join(File.dirname(file), "development.json")), "audit" => audit)]
    end
    final = choices.fetch("checkpoints").to_h do |identity, checkpoint|
      result = sealed(File.join(root, "lockbox", "#{identity}.json"))
      raise "Lockbox weights changed" unless result.fetch("weights_sha256") == checkpoint.fetch("weights_sha256")
      audit = validated_audit(root, identity, "lockbox")
      raise "Final audit weights mismatch" unless audit.fetch("weights_sha256") == checkpoint.fetch("weights_sha256")
      [identity, { "weights_sha256" => checkpoint.fetch("weights_sha256"), "views" => result.fetch("views").transform_values { |metrics| DecisionScaling.compact_metrics(metrics) }, "audit" => audit }]
    end
    archive = protocol.fetch("source_sha256").all? { |file, digest| sha(File.join(root, "source-frozen", file)) == digest }
    raise "Original source archive changed" unless archive
    value = { "status" => "completed", "protocol_sha256" => sha(File.join(root, "protocol.json")), "selection" => decision,
      "confirmation_passed" => choices.fetch("confirmation_passed"), "actual_cuda_updates" => traces.values.sum, "ceiling" => CEILING,
      "trace_diagnostics" => diagnostics, "endpoints" => endpoints, "lockbox" => final,
      "baselines" => [1337, 2027].to_h { |seed| [seed, sealed(File.join(root, "baseline-#{seed}.json")).except("checkpoint").merge("development" => sealed(File.join(root, "baseline-#{seed}.json")).fetch("development").transform_values { |metrics| DecisionScaling.compact_metrics(metrics) },
        "audit" => validated_audit(root, "baseline-#{seed}", "development"))] },
      "data_audit" => protocol.fetch("data_audit").except("historical_files").merge("historical_files_count" => protocol.dig("data_audit", "historical_files").size),
      "isolation_correction" => protocol.fetch("isolation_correction").merge("replay_audit" => JSON.parse(File.read(File.join(root, "isolation-correction/family-replay-audit.json"))),
        "numerical_replay" => JSON.parse(File.read(File.join(root, "isolation-correction/prefix-replay-numerical-audit.json")))),
      "source_archive_verified" => archive, "lockbox_opening" => sealed(File.join(root, "lockbox-opened.json")),
      "paired_error_analysis" => paired_error_analysis(root, choices),
      "read_only_helper_sha256" => %w[event_generalization.rb event_generalization/audit.rb event_generalization/report.rb].to_h { |name| [name, sha(File.join("benchmarks/decision", name))] },
      "release_changed" => false, "scope" => protocol.fetch("scope") }
    value["development_replay"] = File.exist?(File.join(root, "replay-marker.json"))
    value["replay_provenance"] = sealed(File.join(root, "replay-marker.json")) if value.fetch("development_replay")
    reproduction = File.join(root, "reproduced-isolation-audit/comparison.json")
    value["versioned_exclusion_reproduction"] = JSON.parse(File.read(reproduction)) if File.exist?(reproduction)
    value["input_pipeline_audit"] = sealed(File.join(root, "isolation-correction/input-pipeline-audit.json"))
    value["factorial_data_path_equivalence"] = JSON.parse(File.read(File.join(root, "isolation-correction/data-path-equivalence.json")))
    if File.exist?(File.join(root, "crossed-template-plan.json"))
      diagnostic_plan = sealed(File.join(root, "crossed-template-plan.json"))
      raise "Crossed diagnostic registration changed" unless diagnostic_plan.fetch("protocol_sha256") == sha(File.join(root, "protocol.json")) && diagnostic_plan.fetch("selection_sha256") == sha(File.join(root, "selection.json"))
      raise "Crossed diagnostic source changed" unless diagnostic_plan.fetch("audit_source_sha256") == sha("benchmarks/decision/event_generalization/audit.rb")
      raise "Crossed diagnostic registered after opening" unless Time.iso8601(diagnostic_plan.fetch("at")) < Time.iso8601(value.fetch("lockbox_opening").fetch("at"))
      value["crossed_template_plan"] = diagnostic_plan
    end
    value["crossed_template_diagnostic"] = Dir[File.join(root, "crossed-template-diagnostic/*.json")].reject { |file| file.end_with?(".sha256.json") }.to_h do |file|
      identity = File.basename(file, ".json")
      diagnostic = sealed(file)
      raise "Crossed diagnostic checkpoint mismatch" unless diagnostic.fetch("weights_sha256") == choices.fetch("checkpoints").fetch(identity).fetch("weights_sha256")
      diagnostic.fetch("views").each do |view, cell|
        raise "Crossed diagnostic predictions changed" unless sha(File.join(root, "crossed-template-diagnostic", "#{identity}-#{view}.jsonl")) == cell.fetch("predictions_sha256")
      end
      [identity, diagnostic]
    end
    value.fetch("read_only_helper_sha256")["event_generalization/family_replay.rb"] = sha("benchmarks/decision/event_generalization/family_replay.rb")
    if decision.fetch("improving")
      winner = decision.fetch("winner")
      value["lockbox_direction_by_seed"] = [1337, 2027].to_h do |seed|
        candidates = ["reference", winner].map do |variant|
          endpoint = sealed(File.join(root, "arms", "#{variant}-#{seed}", "result.json"))
          panel = sealed(File.join(root, "lockbox", "#{variant}-#{seed}.json"))
          { "development" => panel.fetch("views"), "regression" => endpoint.fetch("regression") }
        end
        [seed, direction?(candidates.last, candidates.first)]
      end
      value["lockbox_direction_scope"] = "The unchanged directional rule applied to fixed final-panel views plus the unchanged old regression scores. No reselection or optimization follows; independent-transfer wording requires inspecting these results, not only development confirmation."
    end
    write(File.join(root, "report.json"), value)
    write("docs/decision/v08-results.json", value)
    puts "Completed v08 report: #{endpoints.size} training endpoints, #{traces.values.sum} actual CUDA updates, confirmation=#{choices.fetch('confirmation_passed').inspect}"
    value
  end

  def validated_audit(root, identity, split)
    value = sealed(File.join(root, "audit", "#{identity}-#{split}.json"))
    rows = DecisionRelational.rows(File.join(root, "audit", identity, "#{split}-expressions-subset.jsonl")).to_h { |row| [row.fetch("id"), row] }
    value.fetch("public_interface").each do |device, checks|
      checks.each do |check|
        guess = check.fetch("probabilities").max_by { |_, probability| probability }.first
        raise "Public/batched candidate mapping mismatch" unless guess == rows.fetch(check.fetch("id")).fetch("#{device}_prediction")
      end
    end
    value.fetch("views").each do |view, cell|
      file = File.join(root, "audit", identity, "#{split}-#{view}-subset.jsonl")
      raise "Audited subset predictions changed" unless sha(file) == cell.fetch("predictions_sha256")
    end
    value.merge("public_predictions_match_same_device_batch" => true)
  end

  def retention_diagnostics(file)
    history = JSON.parse(File.read(file))
    %w[probe probe-augmented].to_h do |name|
      consecutive = 0
      checks = history.map do |row|
        passed = row.fetch("probes").fetch(name).fetch("by_language").values.all? do |cell|
          cell.fetch("balanced") >= 0.95 && cell.fetch("actor_binding_all_correct") >= 0.85 && cell.fetch("event_binding_all_correct") >= 0.85
        end
        consecutive = passed ? consecutive + 1 : 0
        { "step" => row.fetch("step"), "passed" => passed, "consecutive" => consecutive }
      end
      [name, { "checks" => checks, "two_consecutive_passes" => checks.any? { |row| row.fetch("consecutive") >= 2 },
        "scope" => "Recorded post-correction fixed probes only. Diagnostic 95% balanced / 85% actor/event pairs, not a training stop or production gate; full original-training metrics are separate." }]
    end
  end

  def paired_error_analysis(root, choices)
    comparisons = []
    [1337, 2027].each do |seed|
      reference = "reference-#{seed}"
      next unless File.exist?(File.join(root, "arms", reference, "result.json"))
      comparisons << ["development", "baseline-#{seed}", reference]
      if choices.fetch("checkpoints").key?(reference)
        comparisons << ["lockbox", "baseline-#{seed}", reference]
      end
      Dir[File.join(root, "arms", "*-#{seed}", "result.json")].each do |file|
        candidate = File.basename(File.dirname(file))
        next if candidate == reference
        comparisons << ["development", reference, candidate]
        comparisons << ["lockbox", reference, candidate] if choices.fetch("checkpoints").key?(candidate)
      end
    end
    comparisons.to_h do |split, before, after|
      views = Catalog::VIEWS.to_h do |view|
        left = DecisionRelational.rows(File.join(root, "predictions", before, "#{split}-#{view}.jsonl"))
        right = DecisionRelational.rows(File.join(root, "predictions", after, "#{split}-#{view}.jsonl")).to_h { |row| [row.fetch("id"), row] }
        raise "Paired analysis input mismatch" unless left.size == right.size && left.all? { |row| right.fetch(row.fetch("id")).values_at("state", "question", "target") == row.values_at("state", "question", "target") }
        axes = %w[actor event].to_h do |axis|
          selected = left.select { |row| row.dig("world", "mixed") && row.fetch("source").downcase.end_with?(axis) }
          languages = selected.group_by { |row| row.fetch("language") }.transform_values do |rows|
            counts = { "both_correct" => 0, "corrected_pairs" => 0, "lost_pairs" => 0, "neither_correct" => 0 }
            world_changes = Hash.new(0)
            examples = []
            pairs = rows.group_by { |row| [row.fetch("state"), row.dig("world", "assertion")] }
            pairs.each_value do |pair|
              raise "Paired analysis group invalid" unless pair.size == 2 && pair.map { |row| row.fetch("target") }.sort == %w[no yes]
              a = pair.all? { |row| row.fetch("prediction") == row.fetch("target") }
              b = pair.all? { |row| right.fetch(row.fetch("id")).fetch("prediction") == row.fetch("target") }
              category = a ? (b ? "both_correct" : "lost_pairs") : (b ? "corrected_pairs" : "neither_correct")
              counts[category] += 1
              world_changes[pair.first.fetch("group_id")] += (b ? 1 : 0) - (a ? 1 : 0)
              next if b || examples.size >= 2
              examples << pair.map { |row| row.slice("id", "state", "question", "target").merge("before" => row.fetch("prediction"), "after" => right.fetch(row.fetch("id")).fetch("prediction")) }
            end
            counts.merge("pairs" => pairs.size, "worlds" => world_changes.size, "worlds_improved" => world_changes.values.count(&:positive?),
              "worlds_worsened" => world_changes.values.count(&:negative?), "worlds_tied" => world_changes.values.count(&:zero?), "fixed_first_remaining_failures" => examples)
          end
          [axis, languages]
        end
        [view, axes]
      end
      ["#{split}:#{before}->#{after}", views]
    end
  end

  def plot(root)
    verify(root)
    directory = File.join(root, "plot")
    FileUtils.mkdir_p(directory)
    series = Dir[File.join(root, "arms/*/development.json")].sort.map.with_index do |file, index|
      points = JSON.parse(File.read(file)).select { |row| !row.fetch("development_subset").empty? }.map do |row|
        event_min = ->(view) { row.fetch("development_subset").fetch(view).fetch("by_language").values.map { |cell| cell.fetch("event_binding_all_correct") }.min }
        [row.fetch("step"), row.dig("probes", "probe", "worst_language_balanced"), row.dig("probes", "probe-augmented", "worst_language_balanced"),
          event_min.call("events"), event_min.call("expressions")].join(" ")
      end
      next if points.empty?
      path = File.join(directory, "#{index}.dat")
      File.write(path, points.join("\n") + "\n")
      [File.basename(File.dirname(file)), path, index + 1]
    end.compact
    raise "No measured development-subset curves yet" if series.empty?
    endpoints = Dir[File.join(root, "arms/*/result.json")].sort_by do |file|
      value = sealed(file)
      [value.fetch("seed"), VARIANTS.index(value.fetch("variant"))]
    end.map do |file|
      result = sealed(file)
      values = %w[events expressions].flat_map do |view|
        Catalog::LANGUAGES.map { |language| result.dig("development", view, "by_language", language, "event_binding_all_correct") }
      end
      [JSON.generate(File.basename(File.dirname(file))), *values].join(" ")
    end
    endpoint_file = File.join(directory, "complete-development-endpoints.dat")
    File.write(endpoint_file, endpoints.join("\n") + "\n")
    curves = ->(column) { series.map { |name, file, color| "#{JSON.generate(file)} using 1:#{column} with linespoints lc #{color} title #{JSON.generate(name)}" }.join(", ") }
    output = File.expand_path("docs/images/decision-v08-event-generalization.png")
    script = <<~GP
      set terminal pngcairo size 1600,1700 font "Sans,11"
      set output #{JSON.generate(output)}
      set multiplot layout 3,2 margins 0.07,0.98,0.10,0.94 spacing 0.08,0.09 title "Event/expression generalization: learning diagnostics and complete development endpoints"
      set grid
      set xlabel "Additional optimizer updates"
      set yrange [0:1.05]
      set key bottom right opaque maxrows 3 font ",9"
      set title "Original-training probe: worst-language balanced accuracy"
      plot #{curves.call(2)}
      unset key
      set title "Fixed augmented-training probe: worst-language balanced accuracy"
      plot #{curves.call(3)}
      set title "New-event development subset: worst-language event pairs"
      plot #{curves.call(4)}
      set title "New-expression development subset: worst-language event pairs"
      plot #{curves.call(5)}
      set style data histograms
      set style histogram clustered gap 1
      set style fill solid 0.8 border -1
      set boxwidth 0.85
      set autoscale x
      set xtics rotate by -25 font ",9"
      set xlabel "Fixed 6000-step endpoint; 128 pairs / 16 event worlds per cell"
      set key top left opaque horizontal font ",9"
      set title "Complete new-event development: event pairs, EN and ZH"
      plot #{JSON.generate(endpoint_file)} using 2:xtic(1) lc rgb "#377eb8" title "English", '' using 3 lc rgb "#ff7f00" title "Chinese"
      set title "Complete new-expression development: event pairs, EN and ZH"
      plot #{JSON.generate(endpoint_file)} using 4:xtic(1) lc rgb "#377eb8" title "English", '' using 5 lc rgb "#ff7f00" title "Chinese"
      unset multiplot
    GP
    path = File.join(directory, "figure.gp")
    File.write(path, script)
    raise "v08 plotting failed" unless system("gnuplot", path)
    puts "Wrote #{output}"
    plot_lockbox(root)
  end

  def plot_lockbox(root)
    directory = File.join(root, "plot")
    return unless File.exist?(File.join(root, "final-choices.json"))
    choices = sealed(File.join(root, "final-choices.json"))
    return unless choices.fetch("checkpoints").keys.all? { |identity| File.exist?(File.join(root, "lockbox", "#{identity}.json")) }
    rows = choices.fetch("checkpoints").keys.map do |identity|
      result = sealed(File.join(root, "lockbox", "#{identity}.json"))
      cells = %w[events expressions].flat_map do |view|
        Catalog::LANGUAGES.map { |language| result.dig("views", view, "by_language", language, "event_binding_all_correct") }
      end
      [JSON.generate(identity), *cells].join(" ")
    end
    data = File.join(directory, "fixed-lockbox-endpoints.dat")
    File.write(data, rows.join("\n") + "\n")
    output = File.expand_path("docs/images/decision-v08-lockbox.png")
    script = <<~GP
      set terminal pngcairo size 1600,740 font "Sans,12"
      set output #{JSON.generate(output)}
      set multiplot layout 1,2 margins 0.06,0.98,0.27,0.88 spacing 0.08,0.08 title "Fixed final panel: event full-pair correctness (256 pairs / 32 event worlds per cell)"
      set grid ytics
      set yrange [0:1.05]
      set ylabel "Both opposite-label questions correct"
      set style data histograms
      set style histogram clustered gap 1
      set style fill solid 0.8 border -1
      set boxwidth 0.85
      set xtics rotate by -30 font ",10"
      set key top left opaque horizontal
      set title "New event meanings"
      plot #{JSON.generate(data)} using 2:xtic(1) lc rgb "#377eb8" title "English", '' using 3 lc rgb "#ff7f00" title "Chinese"
      set title "New expression templates"
      plot #{JSON.generate(data)} using 4:xtic(1) lc rgb "#377eb8" title "English", '' using 5 lc rgb "#ff7f00" title "Chinese"
      unset multiplot
    GP
    path = File.join(directory, "lockbox-figure.gp")
    File.write(path, script)
    raise "v08 final-panel plotting failed" unless system("gnuplot", path)
    puts "Wrote #{output}"
  end
end
