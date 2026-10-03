module DecisionJudgment
  module_function

  def runtime(root, name)
    verify_opened(root)
    policy = JSON.parse(File.read(File.join(root, "calibration.json"))).fetch(name)
    runtime_checkpoint(root, name, policy.fetch("checkpoint"))
  end

  def runtime_checkpoint(root, name, path)
    result = DecisionRelease.runtime_check(root, path)
    predictor = EasyAI::Decision::Predictor.load(path, device: :auto, candidate_chunk_size: 1)
    rows = panel(root, "validation", "joint").group_by { |row| row.fetch("language") }.values.map(&:first)
    checks = rows.map do |row|
      options = row.fetch("options").each_with_index.map { |option, index| { id: index, text: option.fetch("text") } }
      request = { state: row.fetch("state"), question: row.fetch("question"), options: options }
      first = predictor.probabilities(**request)
      second = predictor.probabilities(**request.merge(options: options.reverse))
      json = JSON.parse(JSON.generate(first))
      delta = first.fetch("probabilities").keys.map { |id| (first.fetch("probabilities").fetch(id) - second.fetch("probabilities").fetch(id)).abs }.max
      { "json_keys" => json.fetch("probabilities").keys.sort, "probability_sum" => first.fetch("probabilities").values.sum, "permutation_delta" => delta }
    end
    memory = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096)
    samples = Array.new(5) do
      rows.cycle.take(10).each { |row| predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options")) }
      GC.start
      memory.check_budget!(predictor.device)
    end
    result["checks"] = checks
    result["memory_mib"] = samples
    result["scope"] = "Two bilingual development requests with three candidates; warm cached single-call latency smoke, not throughput or semantic acceptance"
    result["passed"] &&= checks.all? { |check| check.fetch("json_keys") == %w[0 1 2] && check.fetch("permutation_delta") < 1e-4 && (check.fetch("probability_sum") - 1).abs < 1e-6 } && (predictor.device != "cuda" || samples.max - samples.min <= 64)
    write(File.join(root, "runtime-#{name}.json"), result)
    puts "Runtime #{name}: passed=#{result.fetch('passed')} memory=#{samples}"
  end

  def bootstrap(rows, profile, samples: 300)
    selected = rows.select { |row| row.fetch("view") == "canonical" && row.dig("world", "profile") == profile }
    groups = selected.group_by { |row| row.fetch("group_id") }.values
    rng = Random.new(1704)
    draws = Array.new(samples) do
      # Bootstrap complete bilingual frames; original rows/labels stay paired.
      resampled = Array.new(groups.size) { groups.sample(random: rng) }.flatten(1)
      recalls = resampled.group_by { |row| row.fetch("language") }.transform_values do |items|
        items.group_by { |row| row.fetch("target") }.values.map do |cell|
          cell.count { |row| row.fetch("options").fetch(row.fetch("logits").each_index.max_by { |index| row.fetch("logits")[index] }).fetch("id") == row.fetch("target") }.fdiv(cell.size)
        end.sum.fdiv(profile == "binary" ? 2 : 3)
      end
      recalls.values.min
    end.sort
    { "units" => groups.size, "samples" => samples, "worst_language_balanced_95pct" => [draws[(samples * 0.025).floor], draws[(samples * 0.975).floor]], "scope" => "Whole bilingual semantic frames; views are not independent units" }
  end

  def exposure(root, arm)
    summary = JSON.parse(File.read(File.join(root, "#{arm}-1337/summary.json")))
    rows = raw(File.join(root, "data/#{arm}-train.jsonl"))
    visits = summary.fetch("coverage").fetch("row_visits")
    raise "Exposure mismatch" unless rows.size == visits.size && visits.sum == summary.fetch("examples_seen")
    cells = rows.each_index.group_by { |i| [rows[i].fetch("source"), rows[i].fetch("language"), rows[i].fetch("options").size, rows[i].fetch("target")].join("/") }
      .transform_values { |indexes| { "visits" => indexes.sum { |i| visits[i] }, "unique_rows" => indexes.count { |i| visits[i].positive? }, "available_rows" => indexes.size } }
    { "rows" => rows.size, "updates" => summary.fetch("step"), "visits" => visits.sum, "input_tokens" => summary.fetch("coverage").fetch("input_tokens"), "cells" => cells }
  end

  def publication_failures(root, evaluations, protocol)
    decision = JSON.parse(File.read(File.join(root, "confirmation.json")))
    return ["No development profile qualified for confirmation"] unless decision.fetch("confirmation_required")
    failures = []
    failures << "Replay panels cannot establish independent acceptance" if protocol["replay_of"]
    profile = decision.fetch("profile")
    name = "#{decision.fetch('winner')}-#{profile}"
    %w[canonical held_expression held_options].each do |view|
      failures << "#{view}: acceptance floors failed" unless eligible?(evaluations.fetch(name).fetch(profile).fetch(view).fetch("raw"), profile, protocol.fetch("publication"))
    end
    failures << "Confirmation did not retain profile floors" unless evaluations["confirmation"] && eligible?(evaluations.fetch("confirmation").fetch(profile).fetch("canonical").fetch("raw"), profile, protocol.fetch("publication"))
    failures << "Independent reviewed natural acceptance unavailable" unless protocol.fetch("natural_acceptance_available")
    [name, "confirmation"].each do |model_name|
      next unless evaluations[model_name]
      %w[en-US zh-CN].each do |language|
        actual = evaluations.fetch(model_name).dig("natural", profile, "by_language", language, "accuracy")
        parent = evaluations.fetch("parent").dig("natural", profile, "by_language", language, "accuracy")
        failures << "#{model_name}/#{language}: natural acceptance floor or retention failed" if actual < protocol.dig("publication", "natural_minimum_accuracy") || actual < parent - protocol.dig("publication", "natural_tolerance")
      end
    end
    failures
  end

  def publish(root, policy, profile, report)
    destination = File.expand_path("runs/decision/v0.4-#{profile}-preview")
    raise "Preview already exists" if File.exist?(destination)
    loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
    raise "Accepted weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
    calibration = { "temperature" => policy.fetch("profiles").fetch(profile).fetch("temperature"), "profile" => profile,
      "dataset_sha256" => sha(File.join(root, "data/calibration.jsonl")) }
    checkpoint = EasyAI::Decision::Checkpoint.save(destination, model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), training_state: loaded.fetch(:metadata).fetch("training"), calibration: calibration)
    write(File.join(destination, "capabilities.json"), { "version" => "0.4", "status" => "scoped preview", "profile" => profile,
      "checkpoint" => checkpoint, "scope" => "Chinese/English short explicit records, single-step claims, fixed reviewed candidate wording; binary probabilities cannot detect absent evidence", "report" => report })
    destination
  end

  def report(root)
    fit = JSON.parse(File.read(File.join(root, "fit/result.json")))
    return failed_fit_report(root, fit) unless fit.fetch("passed")
    protocol = verify_opened(root)
    evaluation = JSON.parse(File.read(File.join(root, "evaluation.json")))
    policies = JSON.parse(File.read(File.join(root, "calibration.json")))
    names = policies.keys.reject { |name| %w[parent v03].include?(name) }
    names.each { |name| subprocess(root, "runtime", "--arm", name) }
    runtime = names.to_h { |name| [name, JSON.parse(File.read(File.join(root, "runtime-#{name}.json")))] }
    failures = publication_failures(root, evaluation, protocol)
    failures << "Runtime verification failed" unless runtime.values.all? { |value| value.fetch("passed") }
    summaries = ARMS.to_h do |arm|
      path = File.join(root, "#{arm}-1337/summary.json")
      [arm, File.exist?(path) ? JSON.parse(File.read(path)).except("coverage") : { "not_run" => "Fitting failed" }]
    end
    intervals = evaluation.keys.to_h do |name|
      rows = raw(File.join(root, "predictions-#{name}.jsonl"))
      [name, %w[binary joint].to_h { |profile| [profile, bootstrap(rows, profile)] }]
    end
    value = { "passed" => failures.empty?, "publication_failures" => failures, "evaluations" => evaluation,
      "development" => summaries, "bootstrap" => intervals, "runtime" => runtime,
      "training_exposure" => ARMS.select { |arm| File.exist?(File.join(root, "#{arm}-1337/summary.json")) }.to_h { |arm| [arm, exposure(root, arm)] },
      "protocol_sha256" => sha(File.join(root, "protocol.json")), "scope" => protocol.fetch("scope") }
    if failures.empty?
      choice = JSON.parse(File.read(File.join(root, "confirmation.json")))
      value["preview"] = publish(root, policies.fetch("#{choice.fetch('winner')}-#{choice.fetch('profile')}"), choice.fetch("profile"), value)
    end
    write(File.join(root, "report.json"), value)
    plot(root) if summaries.values.none? { |summary| summary["not_run"] }
    puts JSON.pretty_generate(value.slice("passed", "publication_failures", "preview"))
  end

  def failed_fit_report(root, fit)
    protocol = verify(root)
    raise "Failed fitting must not open acceptance" if File.exist?(File.join(root, "acceptance-opened.json"))
    fit = fitting_diagnostics(root, fit)
    bindings = fitting_binding_history(root)
    runtime_checkpoint(root, "fit", fit.fetch("last_checkpoint"))
    rows = raw(File.join(root, "fit/predictions.jsonl"))
    cells = rows.group_by { |row| [row.fetch("language"), row.dig("world", "query"), row.dig("world", "event"), row.dig("world", "mixed")].join("/") }.transform_values do |items|
      correct = items.count { |row| row.fetch("options").fetch(row.fetch("logits").each_index.max_by { |i| row.fetch("logits")[i] }).fetch("id") == row.fetch("target") }
      { "count" => items.size, "accuracy" => correct.fdiv(items.size) }
    end
    history = raw(File.join(root, "fit/choice/training.jsonl"))
    value = { "passed" => false, "failed_stage" => "Isolated known actor/event counterfactual fitting",
      "publication_failures" => ["Training-only fitting did not meet the frozen near-perfect accuracy and mixed-group checks within 1000 updates"],
      "fit" => fit, "binding_history" => bindings, "training_cells" => cells, "devices" => history.map { |row| row.fetch("device") }.tally,
      "gradient_norm_range" => history.map { |row| row.fetch("gradient_norm_preclip") }.minmax,
      "runtime" => JSON.parse(File.read(File.join(root, "runtime-fit.json"))), "acceptance_opened" => false,
      "protocol_sha256" => sha(File.join(root, "protocol.json")), "scope" => protocol.fetch("scope"),
      "next" => "Inspect failed training cells and verified gradients/masks/parameter references; propose a separately authorized representation comparison with reviewed data fixed. Do not scale the mixture or reopen rules after observing this fit." }
    write(File.join(root, "report.json"), value)
    values = history.map { |row| [row.fetch("step"), row.fetch("train_loss")].join("\t") }
    File.write(File.join(root, "fit/training.tsv"), values.join("\n") + "\n")
    measurements = fit.fetch("measurements").map { |row| [row.fetch("step"), row.fetch("nll"), row.fetch("accuracy"), *row.fetch("by_language").values.map { |cell| cell.fetch("binding_all_correct") }].join("\t") }
    File.write(File.join(root, "fit/measurements.tsv"), measurements.join("\n") + "\n")
    binding_values = bindings.map do |row|
      [row.fetch("step"), *%w[actor event].flat_map { |kind| %w[en-US zh-CN].map { |language| row.fetch("by_language").fetch(language).fetch("#{kind}_binding_all_correct") } }].join("\t")
    end
    File.write(File.join(root, "fit/binding.tsv"), binding_values.join("\n") + "\n")
    script = <<~GNUPLOT
      set terminal pngcairo size 1200,700 enhanced font 'Sans,11'
      set output 'comparison.png'
      set multiplot layout 2,2 title 'Decision v0.4: isolated counterfactual fit (training only; acceptance unopened)'
      set grid
      set xlabel 'Optimizer update'
      set title 'Observed sampled CE (raw)'
      plot 'fit/training.tsv' using 1:2 with lines title 'Training CE'
      set title 'Full fitting-set NLL'
      plot 'fit/measurements.tsv' using 1:2 with linespoints title 'Training-only NLL'
      set yrange [0:1]
      set title 'Full fitting-set accuracy'
      plot 'fit/measurements.tsv' using 1:3 with linespoints title 'Accuracy'
      set title 'Mixed groups, both correct: actor vs event'
      plot 'fit/binding.tsv' using 1:2 with linespoints title 'Actor EN', 'fit/binding.tsv' using 1:3 with linespoints title 'Actor ZH', 'fit/binding.tsv' using 1:4 with linespoints title 'Event EN', 'fit/binding.tsv' using 1:5 with linespoints title 'Event ZH'
      unset multiplot
    GNUPLOT
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Fitting chart failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
    puts JSON.pretty_generate(value.slice("passed", "failed_stage", "devices", "gradient_norm_range", "acceptance_opened"))
  end

  def fitting_binding_history(root)
    path = File.join(root, "fit/binding-history.json")
    return JSON.parse(File.read(path)) if File.exist?(path)
    rows = raw(File.join(root, "data/fit.jsonl"))
    steps = JSON.parse(File.read(File.join(root, "fit/measurements.json"))).map { |value| value.fetch("step") }
    result = steps.map do |step|
      checkpoint = Dir.glob(File.join(root, "fit/choice/checkpoints/step-#{format('%08d', step)}-*/metadata.json")).max
      raise "Fitting checkpoint missing" unless checkpoint
      loaded = EasyAI::Decision::Checkpoint.load(File.dirname(checkpoint))
      value = measure(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, DecisionRobust.device)
      value["step"] = step
      value["weights_sha256"] = loaded.fetch(:weights_fingerprint)
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
      value
    end
    write(path, result)
    result
  end

  def fitting_diagnostics(root, fit)
    loaded = EasyAI::Decision::Checkpoint.load(fit.fetch("last_checkpoint"))
    rows = raw(File.join(root, "data/fit.jsonl"))
    device = DecisionRobust.device
    _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, device)
    records = rows.each_with_index.map { |row, index| row.merge("logits" => logits[index]) }
    DecisionRobust.write_rows(File.join(root, "fit/predictions.jsonl"), records)
    parent = EasyAI::Decision::Checkpoint.load(PARENT)
    change = (loaded.fetch(:model).encoder.embedding.weight.detach.cpu - parent.fetch(:model).encoder.embedding.weight.detach.cpu).abs.max.item
    fit["embedding_max_change"] = change
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config)
    mixed = records.select { |row| row.dig("world", "mixed") }
    groups = mixed.group_by { |row| [row.fetch("group_id"), row.fetch("language"), row.dig("world", "query"), row.dig("world", "assertion"), row.dig("contrast_groups", "question_flip").split(":").last] }
    contrasts = groups.values.map do |pair|
      raise "Incomplete mixed counterfactual" unless pair.size == 2 && pair.map { |row| row.fetch("target") }.sort == %w[no yes]
      ids = pair.map { |row| collator.state_tokens(row.fetch("state")) }
      margins = pair.map { |row| row.fetch("logits")[0] - row.fetch("logits")[1] }
      { "language" => pair.first.fetch("language"), "frame" => pair.first.fetch("group_id"), "same_token_multiset" => ids.first.sort == ids.last.sort,
        "margin_delta" => (margins.first - margins.last).abs, "rows" => pair.map { |row| row.slice("state", "question", "target", "logits") } }
    end
    # Keep the public input contract valid; these constants remove informative content.
    conditions = { "original" => rows, "constant_state" => rows.map { |row| row.merge("state" => "[record withheld]") }, "constant_question" => rows.map { |row| row.merge("question" => "[claim withheld]") } }
    sensitivity = conditions.transform_values { |items| measure(loaded.fetch(:model), loaded.fetch(:tokenizer), items, device) }
    write(File.join(root, "fit/diagnostics.json"), { "embedding_max_change" => change, "mixed_counterfactuals" => contrasts,
      "input_removal" => sensitivity, "scope" => "Training-only diagnostic. Removed inputs retain old labels, not new gold. Equal token multisets plus failed binding suggest weak relational encoding; they do not prove a unique architectural cause." })
    write(File.join(root, "fit/result.json"), fit)
    loaded.fetch(:model).to("cpu")
    GC.start
    fit
  end

  def plot(root)
    ARMS.each do |arm|
      directory = File.join(root, "#{arm}-1337")
      history = raw(File.join(directory, "choice/training.jsonl"))
      File.write(File.join(directory, "training.tsv"), history.map { |row| [row.fetch("step"), row.fetch("train_loss")].join("\t") }.join("\n") + "\n")
      validations = JSON.parse(File.read(File.join(directory, "validation.json")))
      File.write(File.join(directory, "validation.tsv"), validations.map do |row|
        cells = row.fetch("profiles").fetch("binary").fetch("by_language")
        train = row.fetch("training_probe").fetch("by_language")
        [row.fetch("step"), cells.fetch("en-US").fetch("balanced"), cells.fetch("zh-CN").fetch("balanced"),
          cells.fetch("en-US").fetch("binding_all_correct"), cells.fetch("zh-CN").fetch("binding_all_correct"),
          train.fetch("en-US").fetch("balanced"), train.fetch("zh-CN").fetch("balanced"), row.fetch("profiles").fetch("binary").fetch("nll")].join("\t")
      end.join("\n") + "\n")
    end
    script = <<~GNUPLOT
      set terminal pngcairo size 1400,850 enhanced font 'Sans,11'
      set output 'comparison.png'
      set multiplot layout 2,2 title 'Decision v0.4: reviewed supervision / known judgment curriculum'
      set grid
      set xlabel 'Optimizer update'
      set title 'Observed sampled CE (raw, unsmoothed)'
      plot 'control-1337/training.tsv' using 1:2 with lines title 'Control', 'candidate-1337/training.tsv' using 1:2 with lines title 'Reviewed'
      set yrange [0:1]
      set title 'Held-out binary balanced accuracy'
      plot 'control-1337/validation.tsv' using 1:2 with linespoints title 'Control EN', 'control-1337/validation.tsv' using 1:3 with linespoints title 'Control ZH', 'candidate-1337/validation.tsv' using 1:2 with linespoints title 'Reviewed EN', 'candidate-1337/validation.tsv' using 1:3 with linespoints title 'Reviewed ZH'
      set title 'Held-out mixed-truth groups, both correct'
      plot 'control-1337/validation.tsv' using 1:4 with linespoints title 'Control EN', 'control-1337/validation.tsv' using 1:5 with linespoints title 'Control ZH', 'candidate-1337/validation.tsv' using 1:4 with linespoints title 'Reviewed EN', 'candidate-1337/validation.tsv' using 1:5 with linespoints title 'Reviewed ZH'
      set title 'Own-training probe balanced accuracy'
      plot 'control-1337/validation.tsv' using 1:6 with linespoints title 'Control EN', 'control-1337/validation.tsv' using 1:7 with linespoints title 'Control ZH', 'candidate-1337/validation.tsv' using 1:6 with linespoints title 'Reviewed EN', 'candidate-1337/validation.tsv' using 1:7 with linespoints title 'Reviewed ZH'
      unset multiplot
    GNUPLOT
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Plot failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
  end
end
