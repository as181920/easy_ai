module DecisionFactual
  module_function

  def plot(root)
    ARMS.each do |arm|
      directory = File.join(root, "#{arm}-1337")
      history = raw_rows(File.join(directory, "choice/training.jsonl"))
      losses = history.map { |row| row.fetch("train_loss") }
      rows = history.each_with_index.map do |row, index|
        window = losses[[index - 49, 0].max..index]
        [row.fetch("step"), row.fetch("train_loss"), window.sum.fdiv(window.size), row.fetch("gradient_norm_preclip")].join("\t")
      end
      File.write(File.join(directory, "training.tsv"), rows.join("\n") + "\n")
      validation = JSON.parse(File.read(File.join(directory, "validation.json")))
      rows = validation.map do |row|
        [row.fetch("step"), row.fetch("factual_nll"), row.fetch("factual_macro_accuracy"), row.fetch("pair_all_correct"),
          row.fetch("fixed_train_probe").fetch("factual_nll"), row.fetch("embedding_max_change"), row["gpu_process_mib"]].join("\t")
      end
      File.write(File.join(directory, "validation.tsv"), rows.join("\n") + "\n")
    end
    script = <<~GNUPLOT
      set terminal pngcairo size 1400,950 enhanced font 'Sans,11'
      set output 'comparison.png'
      set multiplot layout 2,3 rowsfirst title 'Decision v0.2: CE versus CE + signed pair margin (validation only)'
      set xlabel 'Optimizer update'
      set grid
      set key top right
      set title 'Sampled training loss; 50-update moving average'
      plot 'ce-1337/training.tsv' using 1:2 with lines lc rgb '#bfdbfe' title 'CE raw', \
        'margin-1337/training.tsv' using 1:2 with lines lc rgb '#fed7aa' title 'Margin raw', \
        'ce-1337/training.tsv' using 1:3 with lines lw 2 title 'CE mean', \
        'margin-1337/training.tsv' using 1:3 with lines lw 2 title 'Margin mean'
      set title 'Fixed training-probe factual NLL'
      plot 'ce-1337/validation.tsv' using 1:5 with linespoints title 'CE', 'margin-1337/validation.tsv' using 1:5 with linespoints title 'Margin'
      set title 'Held-out factual NLL'
      plot 'ce-1337/validation.tsv' using 1:2 with linespoints title 'CE', 'margin-1337/validation.tsv' using 1:2 with linespoints title 'Margin'
      set title 'Held-out source/language-macro factual accuracy'
      plot 'ce-1337/validation.tsv' using 1:3 with linespoints title 'CE', 'margin-1337/validation.tsv' using 1:3 with linespoints title 'Margin'
      set title 'Held-out both-correct contrast pairs'
      plot 'ce-1337/validation.tsv' using 1:4 with linespoints title 'CE', 'margin-1337/validation.tsv' using 1:4 with linespoints title 'Margin'
      set title 'Pre-clipping gradient norm'
      plot 'ce-1337/training.tsv' using 1:4 with lines title 'CE', 'margin-1337/training.tsv' using 1:4 with lines title 'Margin'
      unset multiplot
    GNUPLOT
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Plot failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
  end

  def bootstrap(predictions, seed: 1337)
    predictions.group_by { |row| row.fetch("language") }.transform_values do |rows|
      groups = rows.group_by { |row| row.fetch("group_id") }.values.map do |items|
        [items.count { |row| row.fetch("prediction") == row.fetch("target") }, items.size]
      end
      rng = Random.new(seed)
      samples = Array.new(300) do
        draws = Array.new(groups.size) { groups.sample(random: rng) }
        draws.sum(&:first).fdiv(draws.sum(&:last))
      end.sort
      factual = rows.reject { |row| row.fetch("source") == "MASSIVE-Scenario" }
      natural = factual.reject { |row| row.fetch("source").start_with?("Factual-") }
      paired = factual.select { |row| row.dig("contrast_groups", "fact_flip") }.group_by { |row| row.fetch("group_id") }.values.map do |items|
        pairs = items.group_by { |row| row.fetch("contrast_groups").fetch("fact_flip") }.values
        [pairs.count { |pair| pair.all? { |row| row.fetch("prediction") == row.fetch("target") } }, pairs.size]
      end
      { "independent_groups" => groups.size, "accuracy_95_percentile" => [samples[7], samples[292]],
        "factual_source_macro_accuracy_95_percentile" => macro_interval(factual, seed: seed),
        "natural_source_macro_accuracy_95_percentile" => macro_interval(natural, seed: seed),
        "pair_all_correct_95_percentile" => count_interval(paired, seed: seed), "pair_semantic_groups" => paired.size,
        "resamples" => 300, "unit" => "Semantic group resampled with all its decisions; row-weighted accuracy" }
    end
  end

  def count_interval(groups, seed:)
    rng = Random.new(seed)
    samples = Array.new(300) do
      draws = Array.new(groups.size) { groups.sample(random: rng) }
      draws.sum(&:first).fdiv(draws.sum(&:last))
    end.sort
    [samples[7], samples[292]]
  end

  def controlled_slices(predictions, temperature)
    rows = predictions.select { |row| row["world"] }
    calibrator = EasyAI::Decision::Calibrator.new(temperature: temperature)
    %w[phenomenon unseen_event].to_h do |attribute|
      cells = rows.group_by { |row| "#{row.fetch('world').fetch(attribute)}/#{row.fetch('language')}" }.transform_values do |items|
        targets = items.map { |row| row.fetch("options").index { |option| option.fetch("id") == row.fetch("target") } }
        metrics = EasyAI::Decision::Evaluator.metrics(items.map { |row| row.fetch("logits") }, targets, calibrator)
        pairs = items.select { |row| row.dig("contrast_groups", "fact_flip") }.group_by { |row| row.fetch("contrast_groups").fetch("fact_flip") }.values
        metrics.merge("semantic_groups" => items.map { |row| row.fetch("group_id") }.uniq.size,
          "pair_all_correct" => pairs.empty? ? nil : pairs.count { |pair| pair.all? { |row| row.fetch("prediction") == row.fetch("target") } }.fdiv(pairs.size))
      end
      [attribute, cells]
    end
  end

  def macro_interval(rows, seed:)
    # Keep a multi-source semantic group together; macro weighting matches the reported estimand.
    groups = rows.group_by { |row| row.fetch("group_id") }.values.map do |items|
      items.group_by { |row| row.fetch("source") }.transform_values do |source_rows|
        [source_rows.count { |row| row.fetch("prediction") == row.fetch("target") }, source_rows.size]
      end
    end
    rng = Random.new(seed)
    samples = Array.new(300) do
      totals = Hash.new { |hash, source| hash[source] = [0, 0] }
      groups.size.times do
        groups.sample(random: rng).each do |source, (correct, count)|
          totals[source][0] += correct
          totals[source][1] += count
        end
      end
      totals.values.sum { |correct, count| correct.fdiv(count) }.fdiv(totals.size)
    end.sort
    [samples[7], samples[292]]
  end

  def publish(root, name, policy, report)
    destination = File.join(ROOT, "runs/decision/v0.2-factual-preview")
    raise "Preview destination exists" if File.exist?(destination)
    loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
    raise "Evaluated weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
    rows = raw_rows(File.join(root, "data/calibration-clean.jsonl"))
    calibrated = EasyAI::Decision::Checkpoint.save(destination, model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer),
      training_state: loaded.fetch(:metadata).fetch("training"), calibration: { "temperature" => policy.fetch("temperature"),
        "groups" => rows.map { |row| row.fetch("group_id") }.uniq, "dataset_sha256" => Digest::SHA256.file(File.join(root, "data/calibration-clean.jsonl")).hexdigest })
    write(File.join(destination, "capabilities.json"), { "status" => "preview", "version" => "0.2", "selected" => name,
      "checkpoint" => calibrated, "evaluation" => report, "scope" => "Measured English/Chinese factual scoring preview, not universal semantics or business automation" })
    destination
  end

  def report(root)
    protocol = verify(root)
    provenance = verify_panels(root)
    opened = JSON.parse(File.read(File.join(root, "acceptance-opened.json")))
    raise "Calibration policy changed after test opening" unless Digest::SHA256.file(File.join(root, "calibration.json")).hexdigest == opened.fetch("policies_sha256")
    raise "Provenance changed after test opening" unless Digest::SHA256.file(File.join(root, "provenance.json")).hexdigest == opened.fetch("provenance_sha256")
    raise "Confirmation changed after test opening" unless Digest::SHA256.file(File.join(root, "confirmation.json")).hexdigest == opened.fetch("confirmation_sha256")
    confirmation = JSON.parse(File.read(File.join(root, "confirmation.json")))
    evaluations = JSON.parse(File.read(File.join(root, "evaluation.json")))
    policies = JSON.parse(File.read(File.join(root, "calibration.json")))
    audits = ARMS.to_h do |arm|
      summary = JSON.parse(File.read(File.join(root, "#{arm}-1337/summary.json")))
      visits = summary.fetch("coverage").fetch("row_visits")
      raw = raw_rows(File.join(root, "data/train.jsonl"))
      sources = raw.each_index.group_by { |i| raw[i].fetch("source") }.transform_values do |indexes|
        { "visits" => indexes.sum { |i| visits[i] }, "unique_rows" => indexes.count { |i| visits[i] > 0 },
          "unique_groups" => indexes.select { |i| visits[i] > 0 }.map { |i| raw[i].fetch("group_id") }.uniq.size }
      end
      cells = raw.each_index.group_by { |i| "#{raw[i].fetch('source')}/#{raw[i].fetch('language')}" }.transform_values do |indexes|
        { "visits" => indexes.sum { |i| visits[i] }, "unique_rows" => indexes.count { |i| visits[i] > 0 },
          "unique_groups" => indexes.select { |i| visits[i] > 0 }.map { |i| raw[i].fetch("group_id") }.uniq.size }
      end
      [arm, { "examples_seen" => summary.fetch("examples_seen"), "input_tokens" => summary.fetch("coverage").fetch("input_tokens"),
        "sources" => sources, "source_language" => cells, "visits" => visits }]
    end
    raise "Objective arms received different rows" unless audits.fetch("ce").fetch("visits") == audits.fetch("margin").fetch("visits")
    raise "Objective arms saw different input tokens" unless audits.fetch("ce").fetch("input_tokens") == audits.fetch("margin").fetch("input_tokens")
    audits.each_value { |audit| audit.delete("visits") }
    probe = raw_rows(File.join(root, "data/fit.jsonl")).select { |row| row["world"] }.first(48)
    sensitivity = policies.to_h do |model, policy|
      loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
      raise "Evaluated weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
      value = diagnostics(loaded.fetch(:model), loaded.fetch(:tokenizer), probe, "cpu")
      GC.start
      [model, value]
    end
    write(File.join(root, "diagnostics-trained.json"), sensitivity)
    name = confirmation["winner"] && "#{confirmation.fetch('winner')}-1337"
    gates = protocol.fetch("publication")
    failures = []
    if name
      candidate = evaluations.fetch(name)
      reference = evaluations.fetch("release")
      %w[en-US zh-CN].each do |language|
        gain = candidate.fetch("factual_by_language").fetch(language).fetch("source_macro_accuracy") - reference.fetch("factual_by_language").fetch(language).fetch("source_macro_accuracy")
        failures << "#{language}: factual improvement < 3 points" if gain < gates.fetch("minimum_factual_language_gain")
        failures << "#{language}: complete pairs < 80%" if candidate.fetch("pairs_by_language").fetch(language).fetch("all_correct") < gates.fetch("minimum_pair_accuracy")
        natural_gain = candidate.fetch("natural_by_language").fetch(language).fetch("source_macro_accuracy") - reference.fetch("natural_by_language").fetch(language).fetch("source_macro_accuracy")
        failures << "#{language}: natural factual accuracy regressed" if natural_gain < gates.fetch("minimum_natural_language_gain")
        routing_gain = candidate.fetch("routing_by_language").fetch(language).fetch("accuracy") - reference.fetch("routing_by_language").fetch(language).fetch("accuracy")
        failures << "#{language}: routing regressed > 3 points" if routing_gain < -gates.fetch("maximum_routing_regression")
      end
      confirmed = if confirmation.fetch("confirmation_required")
        JSON.parse(File.read(File.join(root, "#{confirmation.fetch('winner')}-2027/summary.json"))).fetch("selected")
                  end
      minimum = confirmation.fetch("reference_validation_macro") + confirmation.fetch("criterion").fetch("minimum_validation_gain")
      failures << "No successful validation confirmation" unless confirmed && confirmed.fetch("metrics").fetch("factual_macro_accuracy") >= minimum
    else
      failures << "Neither arm selected a checkpoint preserving v0.1 routing validation"
    end
    predictions = evaluations.keys.to_h { |model| [model, raw_rows(File.join(root, "predictions-#{model}.jsonl"))] }
    intervals = predictions.transform_values { |rows| bootstrap(rows) }
    controlled = predictions.to_h { |model, rows| [model, controlled_slices(rows, policies.fetch(model).fetch("temperature"))] }
    result = { "selected_by_validation" => name, "publication_failures" => failures, "passed" => failures.empty?,
      "evaluations" => evaluations, "group_bootstrap" => intervals, "training_audit" => audits,
      "train_only_sensitivity" => sensitivity,
      "controlled_slices" => controlled,
      "scope" => protocol.fetch("scope"), "routing_test_scope" => protocol.fetch("routing_test_scope"), "provenance" => provenance }
    runtime_names = ARMS.map { |arm| "#{arm}-1337" }
    runtime_names << "#{confirmation.fetch('winner')}-2027" if confirmation.fetch("confirmation_required")
    runtime = runtime_names.to_h do |model|
      GC.start
      check = DecisionRelease.runtime_check(root, policies.fetch(model).fetch("checkpoint"))
      raise "Evaluated weights changed" unless check.fetch("weights_sha256") == policies.fetch(model).fetch("weights_sha256")
      check["scope"] = "Two bilingual validation examples with their actual candidate counts; warm cached API smoke, not production latency or semantic acceptance"
      [model, check]
    end
    write(File.join(root, "runtime.json"), runtime)
    result["runtime"] = runtime
    failures << "CPU/CUDA runtime smoke failed" if name && !runtime.fetch(name).fetch("passed")
    result["passed"] = failures.empty?
    result["preview"] = publish(root, name, policies.fetch(name), result) if failures.empty?
    write(File.join(root, "report.json"), result)
    plot(root)
    puts JSON.pretty_generate(result.slice("selected_by_validation", "passed", "publication_failures", "preview", "training_audit"))
  end
end
