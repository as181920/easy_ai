module DecisionExpressionRetention
  module_function

  def identities(root)
    values = [1337, 2027].to_h { |seed| ["baseline-#{seed}", { "path" => parent(seed), "weights_sha256" => sha(File.join(parent(seed), "weights.pt")) }] }
    Dir[File.join(root, "arms/*/result.json")].each do |file|
      value = sealed(file)
      values[File.basename(File.dirname(file))] = { "path" => value.fetch("checkpoint"), "weights_sha256" => value.fetch("weights_sha256") }
    end
    values
  end

  def audit(root, identity, split: "development")
    verify(root)
    split = split.to_s
    raise "Unregistered audit split" unless %w[development lockbox].include?(split)
    chosen = identities(root).fetch(identity)
    if split == "lockbox"
      opened = sealed(File.join(root, "lockbox-opened.json"))
      raise "Lockbox choices changed" unless opened.fetch("choices_sha256") == sha(File.join(root, "final-choices.json"))
      sealed(File.join(root, "final-choices.json")).fetch("checkpoints").fetch(identity)
    end
    path = File.join(root, "audit", "#{identity}-#{split}.json")
    return sealed(path) if File.exist?(path)
    raise "Audit weights changed" unless sha(File.join(chosen.fetch("path"), "weights.pt")) == chosen.fetch("weights_sha256")
    loaded = EasyAI::Decision::Checkpoint.load(chosen.fetch("path"))
    views = Catalog::VIEWS.to_h do |view|
      name = "#{split}-#{view}"
      full = DecisionRelational.rows(File.join(root, "predictions", identity, "#{name}.jsonl")).to_h { |row| [row.fetch("id"), row] }
      rows = raw(root, name).select { |row| row.dig("world", "family_id").split(":").last.to_i < 4 }
      cpu = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, "cpu", gc_interval: 8).last
      cuda = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, "cuda", gc_interval: 8).last
      raise "CUDA audit fallback" unless loaded.fetch(:model).parameters.all? { |parameter| parameter.device.to_s.start_with?("cuda") }
      guesses = ->(logits) { rows.each_index.map { |index| rows[index].fetch("options")[logits[index].each_index.max_by { |option| logits[index][option] }].fetch("id") } }
      cpu_predictions, cuda_predictions = guesses.call(cpu), guesses.call(cuda)
      raise "Repeated CUDA inference mismatch" unless rows.each_index.all? { |index| cuda_predictions[index] == full.fetch(rows[index].fetch("id")).fetch("prediction") }
      directory = File.join(root, "audit", identity)
      FileUtils.mkdir_p(directory)
      prediction_file = File.join(directory, "#{name}-subset.jsonl")
      DecisionRobust.write_rows(prediction_file, rows.each_index.map { |index| rows[index].merge("cpu_logits" => cpu[index], "cuda_logits" => cuda[index], "cpu_prediction" => cpu_predictions[index], "cuda_prediction" => cuda_predictions[index]) })
      [view, { "subset_rows" => rows.size, "prediction_differences" => cpu_predictions.zip(cuda_predictions).count { |a, b| a != b },
        "max_absolute_logit_delta" => cpu.zip(cuda).flat_map { |a, b| a.zip(b).map { |left, right| (left - right).abs } }.max,
        "cpu_independent_pairs" => %w[actor event].to_h { |axis| [axis, independent_pairs(rows, cpu_predictions, axis: axis)] },
        "predictions_sha256" => sha(prediction_file) }]
    end
    loaded.fetch(:model).to("cpu")
    checks = %w[cpu cuda].to_h do |device|
      predictor = EasyAI::Decision::Predictor.load(chosen.fetch("path"), device: device, candidate_chunk_size: 1)
      raise "Public inference fell back" unless predictor.device == device
      samples = raw(root, "#{split}-expressions").select { |row| row.dig("world", "mixed") }
        .group_by { |row| [row.fetch("language"), row.fetch("source"), row.dig("world", "form")] }.values.flat_map { |group| group.first(2) }
      values = samples.map do |row|
        a = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options"))
        b = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options").reverse)
        probabilities = a.fetch("probabilities")
        delta = probabilities.keys.map { |key| (probabilities.fetch(key) - b.fetch("probabilities").fetch(key)).abs }.max
        raise "Invalid public probabilities" unless probabilities.values.all?(&:finite?) && (probabilities.values.sum - 1).abs < 1e-6 && delta < 1e-4
        { "id" => row.fetch("id"), "sum" => probabilities.values.sum, "candidate_permutation_delta" => delta,
          "probabilities" => JSON.parse(JSON.generate(a)).fetch("probabilities") }
      end
      predictor.model.to("cpu") if predictor.respond_to?(:model)
      predictor = nil
      GC.start
      [device, values]
    end
    seal(path, { "identity" => identity, "split" => split, "weights_sha256" => chosen.fetch("weights_sha256"), "views" => views, "public_interface" => checks,
      "scope" => "Fixed first four complete worlds per language/view and every held form: 256 canonical or 512 expression/composed rows, including both axes, truth assignments, record orders and claim polarities. Shared forms are not new independent worlds. CPU/CUDA audit is a subset, not full-panel equivalence. Full GPU predictions have independent pair reconstruction." })
  end
end

module DecisionExpressionRetention
  module_function

  def compact_panel(value)
    { "all" => DecisionScaling.compact_metrics(value), "by_form" => value.fetch("by_form").transform_values { |panel| DecisionScaling.compact_metrics(panel) } }
  end

  def checked_audit(root, identity, split)
    DecisionEventGeneralization.validated_audit(root, identity, split)
  end

  def direction_details(candidate, reference)
    a, b = target_cells(candidate), target_cells(reference)
    targets = a.to_h { |key, value| [key, value - b.fetch(key)] }
    balanced = reference.fetch("development").except("composed").flat_map do |view, panel|
      panel.fetch("by_form").flat_map do |form, cell|
        Catalog::LANGUAGES.map do |language|
          ["#{view}/#{form}/#{language}", candidate.dig("development", view, "by_form", form, "by_language", language, "balanced") - cell.dig("by_language", language, "balanced")]
        end
      end
    end.to_h
    old = reference.fetch("regression").flat_map do |view, panel|
      Catalog::LANGUAGES.flat_map do |language|
        %w[balanced actor_binding_all_correct event_binding_all_correct].map do |key|
          ["#{view}/#{language}/#{key}", candidate.dig("regression", view, "by_language", language, key) - panel.dig("by_language", language, key)]
        end
      end
    end.to_h
    { "passed" => direction?(candidate, reference), "candidate_minimum" => a.values.min, "reference_minimum" => b.values.min,
      "minimum_gain" => a.values.min - b.values.min, "target_cell_deltas" => targets,
      "target_pair_regressions_gt_5pp" => targets.select { |_, delta| delta < -0.05 - 1e-12 },
      "balanced_regressions_gt_3pp" => balanced.select { |_, delta| delta < -0.03 - 1e-12 },
      "regression_losses_beyond_guard" => old.select { |key, delta| delta < (key.end_with?("balanced") ? -0.03 : -0.05) - 1e-12 } }
  end

  def exposure_audit(root, result)
    state = JSON.parse(File.read(File.join(result.fetch("checkpoint"), "metadata.json"))).fetch("training")
    actual = state.fetch("rendering_coverage")
    forbidden = JSON.parse(File.read(File.join(root, "forbidden-families.json"))).to_set
    surfaces = JSON.parse(File.read(File.join(root, "forbidden-states.json"))).to_set
    raise "Trained reserved family" unless (actual.fetch("families").keys.to_set & forbidden).empty?
    raise "Trained reserved state" unless (actual.fetch("surface_hashes").keys.to_set & surfaces).empty?
    raise "Bad exposure accounting" unless actual.fetch("event_rows").values.sum == result.fetch("examples_seen") && actual.fetch("axis_form_rows").values.sum == result.fetch("examples_seen")
    raise "Bad fixed endpoint" unless state.fetch("step") == STEPS && result.fetch("step") == STEPS && result.fetch("parameters") == 6_627_841 && result.fetch("examples_seen") == STEPS * 32
    variant = result.fetch("variant")
    canonical_fraction = variant == "no_replay" ? 0.0 : (%w[expressions both].include?(variant) ? 0.5 : 1.0)
    Catalog::LANGUAGES.product(%w[Actor Event]).each do |language, axis|
      rows = actual.fetch("axis_form_rows").select { |key, _| key.start_with?("#{language}/#{axis}/") }
      canonical = rows.fetch("#{language}/#{axis}/0", 0)
      raise "Canonical replay fraction mismatch" unless canonical == rows.values.sum * canonical_fraction
    end
    raise "Held form trained" unless actual.fetch("axis_form_rows").keys.all? { |key| ([0] + Catalog::FORMS.fetch("train")).include?(key.split("/").last.to_i) }
    held = Catalog::POOLS.fetch("development") + Catalog::POOLS.fetch("lockbox")
    raise "Held event trained" unless actual.fetch("event_rows").keys.none? { |key| held.include?(key.split("/").last) }
    { "reserved_family_overlap" => 0, "reserved_state_overlap" => 0, "unique_families" => actual.fetch("families").size,
      "canonical_fraction_per_language_axis" => canonical_fraction, "metadata_sha256" => sha(File.join(result.fetch("checkpoint"), "metadata.json")) }
  end

  def paired_errors(root, split, before, after)
    Catalog::VIEWS.to_h do |view|
      left = DecisionRelational.rows(File.join(root, "predictions", before, "#{split}-#{view}.jsonl"))
      right = DecisionRelational.rows(File.join(root, "predictions", after, "#{split}-#{view}.jsonl"))
      mapping = right.to_h { |row| [row.fetch("id"), row] }
      raise "Paired inputs changed" unless left.size == right.size && left.all? { |row| mapping.fetch(row.fetch("id")).values_at("state", "question", "target") == row.values_at("state", "question", "target") }
      axes = %w[actor event].to_h do |axis|
        selected = left.select { |row| row.dig("world", "mixed") && row.fetch("source").downcase.end_with?(axis) }
        cells = selected.group_by { |row| "#{row.fetch('language')}/form#{row.dig('world', 'form')}" }.transform_values do |rows|
          counts = { "both_correct" => 0, "corrected_pairs" => 0, "lost_pairs" => 0, "neither_correct" => 0 }
          changes = Hash.new(0)
          failures = []
          pairs = rows.group_by { |row| [row.fetch("state"), row.dig("world", "assertion")] }
          pairs.each_value do |pair|
            raise "Invalid paired unit" unless pair.size == 2 && pair.map { |row| row.fetch("target") }.sort == %w[no yes]
            a = pair.all? { |row| row.fetch("prediction") == row.fetch("target") }
            b = pair.all? { |row| mapping.fetch(row.fetch("id")).fetch("prediction") == row.fetch("target") }
            key = a ? (b ? "both_correct" : "lost_pairs") : (b ? "corrected_pairs" : "neither_correct")
            counts[key] += 1
            changes[Catalog.family(pair.first)] += (b ? 1 : 0) - (a ? 1 : 0)
            next if b || failures.size >= 2
            failures << pair.map { |row| row.slice("id", "state", "question", "target").merge("before" => row.fetch("prediction"), "after" => mapping.fetch(row.fetch("id")).fetch("prediction")) }
          end
          counts.merge("pairs" => pairs.size, "worlds" => changes.size, "worlds_improved" => changes.values.count(&:positive?),
            "worlds_worsened" => changes.values.count(&:negative?), "worlds_tied" => changes.values.count(&:zero?), "fixed_first_remaining_failures" => failures)
        end
        [axis, cells]
      end
      [view, axes]
    end
  end

  def report(root)
    protocol = verify(root)
    decision = selection(root)
    choices = sealed(File.join(root, "final-choices.json"))
    opening = sealed(File.join(root, "lockbox-opened.json"))
    raise "Final choices changed" unless opening.fetch("choices_sha256") == sha(File.join(root, "final-choices.json"))
    budget!(root)
    archive = protocol.fetch("source_sha256").all? { |file, digest| sha(File.join(root, "source-frozen", file)) == digest }
    raise "Initial archive changed" unless archive
    endpoints = Dir[File.join(root, "arms/*/result.json")].sort.to_h do |file|
      result = sealed(file)
      identity = File.basename(File.dirname(file))
      raise "Endpoint weights changed" unless sha(File.join(result.fetch("checkpoint"), "weights.pt")) == result.fetch("weights_sha256")
      audit = checked_audit(root, identity, "development")
      value = result.except("checkpoint", "development", "regression", "training_full").merge("target_cells" => target_cells(result), "rank" => rank(result),
        "development" => result.fetch("development").transform_values { |panel| compact_panel(panel) },
        "regression" => result.fetch("regression").transform_values { |panel| compact_panel(panel) }, "training_full" => DecisionScaling.compact_metrics(result.fetch("training_full")),
        "exposure_audit" => exposure_audit(root, result), "audit" => audit)
      [identity, value]
    end
    primary = VARIANTS.to_h { |variant| [variant, sealed(File.join(root, "arms", "#{variant}-1337", "result.json"))] }
    raise "Primary row visits differ" unless primary.values.map { |value| value.fetch("row_visits_sha256") }.uniq.size == 1
    raise "Matched event coverage differs" unless %w[events both no_replay].map { |variant| primary.fetch(variant).dig("rendering_coverage", "event_rows") }.uniq.size == 1
    raise "Matched form coverage differs" unless primary.fetch("expressions").dig("rendering_coverage", "axis_form_rows") == primary.fetch("both").dig("rendering_coverage", "axis_form_rows")
    prediction_files = Dir[File.join(root, "predictions/*/*.json")].reject { |file| file.end_with?(".sha256.json") }
    prediction_files.each do |file|
      panel = sealed(file)
      raise "Full prediction file changed" unless sha(file.sub(/\.json\z/, ".jsonl")) == panel.fetch("predictions_sha256")
    end
    traces = ledger(root)
    traces.each_key do |file|
      File.foreach(File.join(root, file)) do |line|
        row = JSON.parse(line)
        raise "Invalid CUDA trace" unless row.fetch("device") == "cuda" && row.values.grep(Numeric).all?(&:finite?)
      end
    end
    final = choices.fetch("checkpoints").to_h do |identity, checkpoint|
      result = sealed(File.join(root, "lockbox", "#{identity}.json"))
      raise "Final weights differ" unless result.fetch("weights_sha256") == checkpoint.fetch("weights_sha256")
      [identity, { "weights_sha256" => checkpoint.fetch("weights_sha256"), "target_cells" => target_cells({ "development" => result.fetch("views") }),
        "views" => result.fetch("views").transform_values { |panel| compact_panel(panel) }, "audit" => checked_audit(root, identity, "lockbox") }]
    end
    comparisons = {}
    [1337, 2027].each do |seed|
      next unless endpoints.key?("reference-#{seed}")
      comparisons["development:baseline-#{seed}->reference-#{seed}"] = paired_errors(root, "development", "baseline-#{seed}", "reference-#{seed}")
      comparisons["lockbox:baseline-#{seed}->reference-#{seed}"] = paired_errors(root, "lockbox", "baseline-#{seed}", "reference-#{seed}")
      endpoints.keys.grep(/-#{seed}\z/).each do |identity|
        next if identity == "reference-#{seed}"
        comparisons["development:reference-#{seed}->#{identity}"] = paired_errors(root, "development", "reference-#{seed}", identity)
        comparisons["lockbox:reference-#{seed}->#{identity}"] = paired_errors(root, "lockbox", "reference-#{seed}", identity)
        comparisons["regression:reference-#{seed}->#{identity}"] = paired_errors(root, "regression", "reference-#{seed}", identity)
      end
    end
    comparisons["development:both-1337->no_replay-1337"] = paired_errors(root, "development", "both-1337", "no_replay-1337")
    comparisons["lockbox:both-1337->no_replay-1337"] = paired_errors(root, "lockbox", "both-1337", "no_replay-1337")
    comparisons["regression:both-1337->no_replay-1337"] = paired_errors(root, "regression", "both-1337", "no_replay-1337")
    directions = primary.except("reference").transform_values { |result| direction_details(result, primary.fetch("reference")) }
    lockbox_directions = endpoints.keys.reject { |identity| identity.start_with?("reference") }.to_h do |identity|
      seed = identity.split("-").last
      reference = sealed(File.join(root, "arms", "reference-#{seed}", "result.json"))
      candidate = sealed(File.join(root, "arms", identity, "result.json"))
      a = candidate.merge("development" => sealed(File.join(root, "lockbox", "#{identity}.json")).fetch("views"))
      b = reference.merge("development" => sealed(File.join(root, "lockbox", "reference-#{seed}.json")).fetch("views"))
      [identity, direction_details(a, b)]
    end
    value = { "status" => "completed", "protocol_sha256" => sha(File.join(root, "protocol.json")), "selection" => decision,
      "confirmation_passed" => choices.fetch("confirmation_passed"), "actual_cuda_updates" => traces.values.sum, "ceiling" => CEILING,
      "ledger" => traces, "trace_sha256" => traces.keys.to_h { |file| [file, sha(File.join(root, file))] },
      "source_archive_verified" => archive, "verified_full_prediction_files" => prediction_files.size, "final_opening" => opening, "endpoints" => endpoints, "lockbox" => final,
      "primary_direction_details" => directions, "fixed_lockbox_direction_details" => lockbox_directions, "paired_error_analysis" => comparisons,
      "baselines" => [1337, 2027].to_h { |seed| result = sealed(File.join(root, "baseline-#{seed}.json")); [seed, result.except("checkpoint", "development").merge("development" => result.fetch("development").transform_values { |panel| compact_panel(panel) }, "audit" => checked_audit(root, "baseline-#{seed}", "development"))] },
      "data_audit" => protocol.fetch("data_audit").except("historical_files").merge("historical_file_count" => protocol.dig("data_audit", "historical_files").size),
      "preflight" => VARIANTS.to_h { |variant| [variant, sealed(File.join(root, "preflight", variant, "result.json"))] },
      "presentation_sha256" => sha("benchmarks/decision/expression_retention/presentation.rb"), "release_changed" => false,
      "scope" => protocol.fetch("scope"), "metric_layout" => "Each view has all languages and by_form language cells. Repeated forms share world families; target_cells and paired errors retain separate template/axis counts." }
    value["replay"] = sealed(File.join(root, "replay-marker.json")) if File.exist?(File.join(root, "replay-marker.json"))
    value["historical_parent_regression"] = historical_parent_regression(root)
    if File.exist?(File.join(root, "historical-wording-plan.json"))
      value["historical_wording_plan"] = sealed(File.join(root, "historical-wording-plan.json"))
      raise "Historical wording source archive changed" unless sha(File.join(root, "historical-wording-source.rb")) == value.fetch("historical_wording_plan").fetch("source_snapshot_sha256")
      value["historical_wording_audits"] = Dir[File.join(root, "historical-wording-audit/*.json")].reject { |file| file.end_with?(".sha256.json") }.select { |file| File.file?(file + ".sha256.json") }.to_h do |file|
        diagnostic = sealed(file)
        identity = diagnostic.fetch("identity")
        raise "Historical wording fixed weights mismatch" unless diagnostic.fetch("weights_sha256") == final.fetch(identity).fetch("weights_sha256")
        raise "Historical wording plan changed" unless diagnostic.fetch("plan_sha256") == sha(File.join(root, "historical-wording-plan.json"))
        diagnostic.fetch("cases").each do |name, cell|
          next unless cell["predictions_sha256"]
          raw_file = File.join(root, "historical-wording-audit", "#{identity}-#{name}.jsonl")
          raise "Historical wording predictions changed" unless sha(raw_file) == cell.fetch("predictions_sha256")
        end
        [identity, diagnostic]
      end
    end
    value["canonical_render_audit"] = sealed(File.join(root, "canonical-render-audit.json")) if File.exist?(File.join(root, "canonical-render-audit.json"))
    value["data_path_replay"] = Dir[File.join(root, "data-path-replay-*.json")].reject { |file| file.end_with?(".sha256.json") }.to_h { |file| [File.basename(file), sealed(file)] }
    if File.exist?(File.join(root, "supplemental-verification.json"))
      supplemental = sealed(File.join(root, "supplemental-verification.json"))
      supplemental.fetch("sources").each do |file, digest|
        raise "Verification test changed" unless sha(file) == digest && sha(File.join(root, "verification-sources", file)) == digest
      end
      value["supplemental_verification"] = supplemental
    end
    write(File.join(root, "report.json"), value)
    write("docs/decision/v09-results.json", value)
    puts "Completed v09: #{endpoints.size} endpoints, #{final.size} fixed final checkpoints, #{traces.values.sum} CUDA updates; confirmation=#{choices.fetch('confirmation_passed').inspect}"
    value
  end
end

module DecisionExpressionRetention
  module_function

  def plot(root)
    verify(root)
    directory = File.join(root, "plot")
    FileUtils.mkdir_p(directory)
    series = Dir[File.join(root, "arms/*/development.json")].sort.map.with_index do |file, index|
      history = JSON.parse(File.read(file))
      data = history.map { |row| [row.fetch("step"), row.dig("probes", "probe", "worst_language_balanced"), row.dig("probes", "probe-augmented", "worst_language_balanced")].join(" ") }
      path = File.join(directory, "curve-#{index}.dat")
      File.write(path, data.join("\n") + "\n")
      [File.basename(File.dirname(file)), path, index + 1]
    end
    endpoints = Dir[File.join(root, "arms/*/result.json")].sort.map do |file|
      result = sealed(file)
      cells = target_cells(result).values
      [JSON.generate(File.basename(File.dirname(file))), cells.min, cells.sum.fdiv(cells.size)].join(" ")
    end
    dev = File.join(directory, "complete-development.dat")
    File.write(dev, endpoints.join("\n") + "\n")
    choices = sealed(File.join(root, "final-choices.json"))
    final = choices.fetch("checkpoints").keys.to_h do |identity|
      result = sealed(File.join(root, "lockbox", "#{identity}.json"))
      [identity, target_cells({ "development" => result.fetch("views") })]
    end
    final_file = File.join(directory, "complete-final.dat")
    File.write(final_file, final.map { |identity, cells| [JSON.generate(identity), cells.values.min, cells.values.sum.fdiv(cells.size)].join(" ") }.join("\n") + "\n")
    curves = ->(column) { series.map { |name, file, color| "#{JSON.generate(file)} using 1:#{column} with linespoints lc #{color} title #{JSON.generate(name)}" }.join(", ") }
    output = File.expand_path("docs/images/decision-v09-coverage-retention.png")
    script = <<~GP
      set terminal pngcairo noenhanced size 1600,1300 font "Sans,11"
      set output #{JSON.generate(output)}
      set multiplot layout 2,2 margins 0.07,0.98,0.16,0.93 spacing 0.08,0.16 title "Event/wording coverage and canonical replay: probes and fixed endpoint full pairs"
      set grid ytics
      set yrange [0:1.05]
      set xlabel "Additional optimizer updates"
      set key bottom right opaque maxrows 4 font ",9"
      set title "Original-training probe: worst-language balanced accuracy"
      plot #{curves.call(2)}
      unset key
      set title "Globally mapped training probe, form7: worst-language balanced accuracy"
      plot #{curves.call(3)}
      set style data histograms
      set style histogram clustered gap 1
      set style fill solid 0.8 border -1
      set boxwidth 0.85
      set autoscale x
      set xtics rotate by -30 font ",9"
      set xlabel "Fixed checkpoints; 12 separate language/axis/template cells"
      set key top left opaque horizontal
      set title "Complete development: minimum and mean target full pairs"
      plot #{JSON.generate(dev)} using 2:xtic(1) lc rgb "#377eb8" title "Minimum", '' using 3 lc rgb "#ff7f00" title "Mean"
      set title "Fixed final panel: minimum and mean target full pairs"
      plot #{JSON.generate(final_file)} using 2:xtic(1) lc rgb "#377eb8" title "Minimum", '' using 3 lc rgb "#ff7f00" title "Mean"
      unset multiplot
    GP
    path = File.join(directory, "figure.gp")
    File.write(path, script)
    raise "v09 plotting failed" unless system("gnuplot", path)
    puts "Wrote #{output}"
    plot_cells(root, final)
    plot_historical_retention(root)
  end

  def plot_cells(root, final)
    directory = File.join(root, "plot")
    data = final.each_with_index.flat_map do |(_, cells), y|
      cells.values.each_with_index.map { |score, x| "#{x} #{y} #{score}" } + [""]
    end
    path = File.join(directory, "final-cells.dat")
    File.write(path, data.join("\n") + "\n")
    labels = final.values.first.keys.each_with_index.map { |key, index| "#{JSON.generate(key.gsub('en-US', 'EN').gsub('zh-CN', 'ZH'))} #{index}" }.join(", ")
    names = final.keys.each_with_index.map { |key, index| "#{JSON.generate(key)} #{index}" }.join(", ")
    output = File.expand_path("docs/images/decision-v09-final-cells.png")
    script = <<~GP
      set terminal pngcairo noenhanced size 1600,850 font "Sans,11"
      set output #{JSON.generate(output)}
      set lmargin at screen 0.15
      set rmargin at screen 0.90
      set bmargin at screen 0.27
      set tmargin at screen 0.90
      set title "Fixed final full pairs by template, language and axis: shared families across expression forms"
      set xrange [-0.5:#{final.values.first.size - 0.5}]
      set yrange [-0.5:#{final.size - 0.5}]
      set xtics (#{labels}) rotate by -40 font ",10"
      set ytics (#{names})
      set cbrange [0:1]
      set cblabel "Both opposite-label questions correct"
      set palette defined (0 "#d73027", 0.5 "#ffffbf", 1 "#1a9850")
      unset key
      plot #{JSON.generate(path)} using 1:2:3 with image
    GP
    script_file = File.join(directory, "final-cells.gp")
    File.write(script_file, script)
    raise "v09 cell plotting failed" unless system("gnuplot", script_file)
    puts "Wrote #{output}"
  end

  def plot_historical_retention(root)
    parents = historical_parent_regression(root).fetch("parents")
    panels = parents.to_h { |seed, value| ["baseline-#{seed}", value.dig("views", "expressions", "by_form")] }
    Dir[File.join(root, "arms/*/result.json")].sort.each do |file|
      value = sealed(file)
      panels[File.basename(File.dirname(file))] = value.dig("regression", "expressions", "by_form").transform_values { |panel| DecisionScaling.compact_metrics(panel) }
    end
    cells = %w[5 6].product(Catalog::LANGUAGES, %w[actor event])
    rows = panels.each_with_index.flat_map do |(_, forms), y|
      cells.each_with_index.map { |(form, language, axis), x| "#{x} #{y} #{forms.dig(form, language, "#{axis}_binding_all_correct")}" } + [""]
    end
    path = File.join(root, "plot", "historical-retention.dat")
    File.write(path, rows.join("\n") + "\n")
    xlabels = cells.each_with_index.map { |(form, language, axis), index| "#{JSON.generate("form#{form}/#{language == 'en-US' ? 'EN' : 'ZH'}/#{axis}")} #{index}" }.join(", ")
    ylabels = panels.keys.each_with_index.map { |identity, index| "#{JSON.generate(identity)} #{index}" }.join(", ")
    output = File.expand_path("docs/images/decision-v09-historical-retention.png")
    script = <<~GP
      set terminal pngcairo noenhanced size 1400,850 font "Sans,11"
      set output #{JSON.generate(output)}
      set lmargin at screen 0.17
      set rmargin at screen 0.89
      set bmargin at screen 0.23
      set tmargin at screen 0.90
      set title "Observed historical expression retention: original parents and fixed continuations"
      set xrange [-0.5:#{cells.size - 0.5}]
      set yrange [-0.5:#{panels.size - 0.5}]
      set xtics (#{xlabels}) rotate by -35 font ",11"
      set ytics (#{ylabels})
      set cbrange [0:1]
      set cblabel "Full pairs; 256 pairs / 32 shared worlds per cell"
      set palette defined (0 "#d73027", 0.5 "#ffffbf", 1 "#1a9850")
      unset key
      plot #{JSON.generate(path)} using 1:2:3 with image
    GP
    script_file = File.join(root, "plot", "historical-retention.gp")
    File.write(script_file, script)
    raise "Historical retention plotting failed" unless system("gnuplot", script_file)
    puts "Wrote #{output}"
  end
end

module DecisionExpressionRetention
  # Read-only RNG replay; never constructs a model or performs optimization.
  class TrackingExclusions < Set
    attr_reader :hits

    def initialize(values)
      super(values)
      @hits = Hash.new(0)
    end

    def include?(value)
      found = super
      @hits[value] += 1 if found
      found
    end
  end

  module_function

  def audit_data_paths(root, steps: 1000, seed: 1337, include_no_replay: false)
    verify(root)
    variants = include_no_replay ? VARIANTS : %w[reference events expressions both]
    raw_rows = raw(root, "train")
    examples = raw_rows.map { |row| EasyAI::Decision::Data::Example.new(row) }
    units = Hash.new { |hash, key| hash[key] = [] }
    raw_rows.each_index.group_by { |index| row = raw_rows[index]; [row.fetch("language"), row.fetch("source").split("-").last, row.fetch("group_id"), row.fetch("state")] }.each do |(language, axis, _, _), indices|
      raise "Bad replay unit" unless indices.size == 4
      units[[language, axis]] << indices
    end
    forbidden = JSON.parse(File.read(File.join(root, "forbidden-families.json")))
    forbidden_states = JSON.parse(File.read(File.join(root, "forbidden-states.json")))
    legacy_states = JSON.parse(File.read(File.join(DecisionEventGeneralization::ROOT_V07, "forbidden-state-hashes.json"))).to_set
    replicas = variants.to_h do |variant|
      replica = Trainer.allocate
      replica.instance_variable_set(:@variant, variant)
      replica.instance_variable_set(:@augmentation, Augmentation.new(rows: raw_rows))
      replica.instance_variable_set(:@pools, Catalog::LANGUAGES.to_h { |language| [language, EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(language, "canonical")] })
      replica.instance_variable_set(:@families, TrackingExclusions.new(forbidden))
      replica.instance_variable_set(:@states, TrackingExclusions.new(forbidden_states))
      [variant, replica]
    end
    coverage = variants.to_h { |variant| [variant, { "families" => Set.new, "surface_hashes" => Set.new, "event_rows" => Hash.new(0), "axis_form_rows" => Hash.new(0) }] }
    differences = { "reference/expressions" => 0, "events/both" => 0 }
    differences["both/no_replay"] = 0 if include_no_replay
    visits = Array.new(raw_rows.size, 0)
    steps.times do |step|
      rng = Random.new(seed + step * 1009)
      indices = Catalog::LANGUAGES.product(%w[Actor Event]).flat_map { |key| 2.times.flat_map { units.fetch(key).sample(random: rng) } }
      indices.each { |index| visits[index] += 1 }
      indices.each_slice(16).with_index do |batch, micro|
        rendered = replicas.transform_values { |replica| replica.render_batch(batch.map { |index| examples[index] }, seed: (seed + step * 1009 + micro) % (2**31)) }
        rendered.each do |variant, rows|
          rows.each do |row|
            coverage[variant]["event_rows"]["#{row.fetch('language')}/#{row.dig('world', 'query_event')}"] += 1
            coverage[variant]["axis_form_rows"]["#{row.fetch('language')}/#{row.fetch('source').split('-').last}/#{row.dig('world', 'form')}"] += 1
          end
          rows.each_slice(4) do |unit|
            coverage[variant]["families"] << Catalog.family(unit.first)
            coverage[variant]["surface_hashes"] << Catalog.material(unit.first.fetch("state"))
          end
        end
        differences.each_key do |comparison|
          a, b = comparison.split("/")
          differences[comparison] += rendered[a].each_slice(4).zip(rendered[b].each_slice(4)).count { |left, right| Catalog.family(left.first) != Catalog.family(right.first) }
        end
      end
    end
    evidence = variants.to_h do |variant|
      choice = File.join(root, "arms", "#{variant}-#{seed}", "choice")
      file = Dir[File.join(choice, "**/metadata.json")].find { |candidate| JSON.parse(File.read(candidate)).dig("training", "step") == steps }
      raise "Requested replay checkpoint missing" unless file
      state = JSON.parse(File.read(file)).fetch("training")
      actual = state.fetch("rendering_coverage")
      raise "Replayed row visits differ" unless state.dig("coverage", "row_visits") == visits
      %w[families surface_hashes].each { |key| raise "Replayed #{key} differ" unless coverage[variant][key] == actual.fetch(key).keys.to_set }
      %w[event_rows axis_form_rows].each { |key| raise "Replayed #{key} differ" unless coverage[variant][key] == actual.fetch(key) }
      family_hits = replicas[variant].instance_variable_get(:@families).hits
      state_hits = replicas[variant].instance_variable_get(:@states).hits
      [variant, { "metadata_sha256" => sha(file), "family_rejections" => family_hits.values.sum,
        "legacy_exact_state_rejections" => state_hits.sum { |key, count| legacy_states.include?(key) ? count : 0 },
        "reserved_normalized_state_rejections" => state_hits.sum { |key, count| legacy_states.include?(key) ? 0 : count },
        "matched_families" => actual.fetch("families").size, "matched_states" => actual.fetch("surface_hashes").size }]
    end
    path = File.join(root, "data-path-replay-#{steps}-#{seed}.json")
    value = { "steps" => steps, "seed" => seed, "units_per_variant" => steps * 8, "different_renamed_units" => differences, "evidence" => evidence,
      "scope" => "Exact actual metadata agreement for row visits, family/state sets and event/axis/form frequencies. Old exact-state exclusions can reject canonical text but allow another form of a nonreserved family, slightly changing renaming draws. This is a disclosed input-path difference, not a reserved-family leak. No model inference or optimizer updates." }
    File.exist?(path) ? (raise "Replay evidence changed" unless sealed(path) == value) : seal(path, value)
    puts JSON.pretty_generate(value)
    value
  end
end

module DecisionExpressionRetention
  module_function

  def historical_parent_regression(root)
    verify(root)
    path = File.join(root, "historical-parent-regression.json")
    return sealed(path) if File.file?(path)
    values = [1337, 2027].to_h do |seed|
      identity = "baseline-#{seed}"
      original = sealed(File.join(ROOT_V08, "baseline-#{seed}.json"))
      weights = sha(File.join(parent(seed), "weights.pt"))
      raise "Historical parent mismatch" unless original.fetch("weights_sha256") == weights
      final_source = sealed(File.join(ROOT_V08, "lockbox", "#{identity}.json"))
      raise "Historical final weights mismatch" unless final_source.fetch("weights_sha256") == weights
      source_hashes = {}
      views = Catalog::VIEWS.to_h do |view|
        file = File.join(ROOT_V08, "predictions", identity, "lockbox-#{view}.jsonl")
        metadata = sealed(file.sub(/\.jsonl\z/, ".json"))
        raise "Historical predictions changed" unless sha(file) == metadata.fetch("predictions_sha256")
        source_hashes[file] = sha(file)
        logits_by_id = DecisionRelational.rows(file).to_h { |row| [row.fetch("id"), row] }
        if %w[expressions composed].include?(view)
          crossed_file = File.join(ROOT_V08, "crossed-template-diagnostic", "#{identity}-#{view}.jsonl")
          crossed = sealed(File.join(ROOT_V08, "crossed-template-diagnostic", "#{identity}.json"))
          raise "Historical crossed weights differ" unless crossed.fetch("weights_sha256") == weights
          raise "Historical crossed predictions changed" unless sha(crossed_file) == crossed.dig("views", view, "predictions_sha256")
          source_hashes[crossed_file] = sha(crossed_file)
          DecisionRelational.rows(crossed_file).each { |row| logits_by_id["#{row.fetch('id')}:crossed"] = row }
        end
        rows = raw(root, "regression-#{view}")
        raise "Historical rows differ" unless rows.size == logits_by_id.size && rows.all? { |row| logits_by_id.fetch(row.fetch("id")).values_at("state", "question", "target", "options") == row.values_at("state", "question", "target", "options") }
        score = metrics(rows, rows.map { |row| logits_by_id.fetch(row.fetch("id")).fetch("logits") })
        [view, compact_panel(score)]
      end
      [seed, { "weights_sha256" => weights, "views" => views, "prediction_sources_sha256" => source_hashes }]
    end
    seal(path, { "parents" => values, "scope" => "Observed v08 regression scores reconstructed from the exact original v07 parents' saved uncalibrated CUDA logits, with both expression/composed forms crossed. Full text/options/target identity and original prediction/weight hashes verified. No new inference, v09 final predictions, optimization or eligibility changes." })
  end
end

module DecisionExpressionRetention
  module_function

  def historical_wording_audit(root, identity)
    verify(root)
    plan = sealed(File.join(root, "historical-wording-plan.json"))
    raise "Historical wording protocol changed" unless plan.fetch("protocol_sha256") == sha(File.join(root, "protocol.json"))
    if identity.start_with?("baseline-")
      seed = Integer(identity.split("-").last)
      raise "Unknown parent" unless [1337, 2027].include?(seed)
      chosen = { "path" => parent(seed), "weights_sha256" => sha(File.join(parent(seed), "weights.pt")) }
      historical_parent_regression(root)
      diagonals = sealed(File.join(root, "historical-parent-regression.json")).dig("parents", seed.to_s, "views", "expressions", "by_form")
      source = File.join(root, "historical-parent-regression.json")
    else
      source = File.join(root, "arms", identity, "result.json")
      result = sealed(source)
      chosen = { "path" => result.fetch("checkpoint"), "weights_sha256" => result.fetch("weights_sha256") }
      diagonals = result.dig("regression", "expressions", "by_form").transform_values { |panel| DecisionScaling.compact_metrics(panel) }
    end
    raise "Fixed historical audit weights changed" unless sha(File.join(chosen.fetch("path"), "weights.pt")) == chosen.fetch("weights_sha256")
    directory = File.join(root, "historical-wording-audit")
    path = File.join(directory, "#{identity}.json")
    if File.file?(path)
      previous = sealed(path)
      raise "Historical wording checkpoint mismatch" unless previous.fetch("weights_sha256") == chosen.fetch("weights_sha256")
      return previous
    end
    rows = raw(root, "regression-expressions").select { |row| row.dig("world", "form") == 5 }
    raise "Wrong historical family allocation" unless rows.size == 4096
    cases = [5, 6].to_h { |form| ["state#{form}-query#{form}", { "rows" => rows.size, "metrics" => diagonals.fetch(form.to_s), "source_sha256" => sha(source) }] }
    loaded = EasyAI::Decision::Checkpoint.load(chosen.fetch("path"))
    [[5, 6], [6, 5]].each do |state_form, query_form|
      name = "state#{state_form}-query#{query_form}"
      changed = rows.map do |row|
        rendered = Catalog.render(row, form: state_form)
        question = Catalog.render(row, form: query_form).fetch("question")
        rendered.merge("question" => question, "world" => rendered.fetch("world").merge("state_form" => state_form, "query_form" => query_form))
      end
      changed.each { |row| Corpus.gold!(row) }
      _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), changed, "cuda", gc_interval: 8)
      raise "Historical wording fell back" unless loaded.fetch(:model).parameters.all? { |parameter| parameter.device.to_s.start_with?("cuda") }
      value = metrics(changed, logits)
      file = File.join(directory, "#{identity}-#{name}.jsonl")
      FileUtils.mkdir_p(directory)
      DecisionRobust.write_rows(file, changed.each_index.map do |index|
        prediction = changed[index].fetch("options")[logits[index].each_index.max_by { |option| logits[index][option] }].fetch("id")
        changed[index].merge("prediction" => prediction, "logits" => logits[index])
      end)
      cases[name] = { "rows" => changed.size, "metrics" => DecisionScaling.compact_metrics(value), "predictions_sha256" => sha(file) }
      puts "Historical wording #{identity}/#{name}: #{value.fetch('worst_language_balanced').round(4)}"
      $stdout.flush
    end
    loaded.fetch(:model).to("cpu")
    seal(path, { "identity" => identity, "weights_sha256" => chosen.fetch("weights_sha256"), "plan_sha256" => sha(File.join(root, "historical-wording-plan.json")),
      "helper_sha256" => sha("benchmarks/decision/expression_retention/presentation.rb"), "cases" => cases,
      "scope" => "Post-hoc state5/6 by question5/6 factorial on the same observed v08 expression families. Diagonal scores reused from verified saved logits; off-diagonal CUDA predictions independently reconstruct both binding axes. Each language/axis cell has256 pairs sharing32 worlds. No new independent worlds, v09 final predictions, selection changes or optimizer updates." })
  end
end
