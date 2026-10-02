module DecisionRobust
  module_function

  def runtime(root, name)
    verify_opened(root)
    policy = JSON.parse(File.read(File.join(root, "calibration.json"))).fetch(name)
    loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
    raise "Runtime weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
    loaded = nil
    GC.start
    value = DecisionRelease.runtime_check(root, policy.fetch("checkpoint"))
    samples = raw_rows(File.join(root, "data/validation.jsonl")).select { |row| row.dig("world", "version") == 3 }
      .group_by { |row| row.fetch("language") }.values.map(&:first)
    predictor = EasyAI::Decision::Predictor.load(policy.fetch("checkpoint"), device: "cuda", candidate_chunk_size: 1)
    memory = EasyAI::Runtime::DevicePolicy.new(requested: "cuda", budget_mib: 4096)
    checks = samples.map do |row|
      numeric = row.fetch("options").each_with_index.map { |option, index| { id: index, text: option.fetch("text") } }
      input = { state: row.fetch("state"), question: row.fetch("question"), options: numeric }
      values = predictor.probabilities(**input)
      reordered = predictor.probabilities(**input.merge(options: numeric.reverse))
      raise "Invalid probability JSON" unless JSON.parse(JSON.generate(values)).fetch("probabilities").keys.sort == %w[0 1 2]
      delta = values.fetch("probabilities").keys.map { |id| (values.fetch("probabilities").fetch(id) - reordered.fetch("probabilities").fetch(id)).abs }.max
      { "language" => row.fetch("language"), "chunk_size" => 1, "permutation_max_delta" => delta,
        "probability_sum" => values.fetch("probabilities").values.sum, "device" => predictor.device }
    end
    samples.cycle.take(10).each { |row| predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options")) }
    measurements = [memory.check_budget!(predictor.device)]
    4.times do
      samples.cycle.take(10).each { |row| predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options")) }
      GC.start
      measurements << memory.check_budget!(predictor.device)
    end
    value["chunked_numeric_id_checks"] = checks
    value["repeated_inference_memory_mib"] = measurements
    value["passed"] &&= predictor.device == "cuda" && checks.all? { |check| check.fetch("permutation_max_delta") < 1e-4 && (check.fetch("probability_sum") - 1).abs < 1e-6 } &&
      measurements.max - measurements.min <= 64
    value["scope"] = "Separate Ruby process, two bilingual development requests, warm cached calls; local latency smoke rather than throughput benchmark"
    write(File.join(root, "runtime-#{name}.json"), value)
    puts JSON.pretty_generate(value)
  end

  def exposure(root, arm)
    summary = JSON.parse(File.read(File.join(root, "#{arm}-1337/summary.json")))
    rows = raw_rows(File.join(root, "data/#{arm}-train.jsonl"))
    visits = summary.fetch("coverage").fetch("row_visits")
    raise "Coverage counters inconsistent" unless visits.size == rows.size && visits.sum == summary.fetch("examples_seen")
    cells = rows.each_index.group_by do |i|
      row = rows[i]
      world = row["world"] || {}
      [row.fetch("source"), row.fetch("language"), row.fetch("target"), world["queried_position"], world["variant"]].join("/")
    end.transform_values do |indexes|
      { "visits" => indexes.sum { |i| visits[i] }, "unique_rows" => indexes.count { |i| visits[i].positive? },
        "unique_groups" => indexes.select { |i| visits[i].positive? }.map { |i| rows[i].fetch("group_id") }.uniq.size }
    end
    { "rows" => rows.size, "examples_seen" => summary.fetch("examples_seen"), "input_tokens" => summary.fetch("coverage").fetch("input_tokens"), "cells" => cells }
  end

  def position_references(root)
    path = File.join(root, "data/test.jsonl")
    rows = raw_rows(path).reject { |row| row.fetch("source") == "MASSIVE-Scenario" }
    values = %w[first last uniform].to_h do |kind|
      cells = rows.group_by { |row| [row.fetch("language"), row.fetch("source")] }.transform_values do |items|
        items.sum do |row|
          options = row.fetch("options")
          kind == "uniform" ? 1.0 / options.size : options.fetch(kind == "first" ? 0 : -1).fetch("id").to_s == row.fetch("target").to_s ? 1.0 : 0.0
        end.fdiv(items.size)
      end
      languages = cells.group_by { |(language, _), _| language }.transform_values { |items| items.sum(&:last).fdiv(items.size) }
      [kind, { "factual_macro_accuracy" => languages.values.sum.fdiv(languages.size), "by_language" => languages,
        "by_source_language" => cells.transform_keys { |key| key.join("/") } }]
    end
    { "dataset_sha256" => sha(path), "values" => values,
      "scope" => "Reporting-only references after policy freeze; no selection/rule change. First/last select only position, never source or gold; uniform is analytical expected accuracy." }
  end

  def publication_failures(root, evaluations, protocol)
    decision = JSON.parse(File.read(File.join(root, "confirmation.json")))
    failures = []
    unless decision.fetch("confirmation_required")
      return ["No candidate with retained routing and non-catastrophic slices improved over both parent and control on development"]
    end
    seed = protocol.fetch("confirmation").fetch("seed")
    confirmation = JSON.parse(File.read(File.join(root, "candidate-#{seed}/summary.json"))).fetch("selected")
    minimum = [decision.fetch("parent_macro") + protocol.fetch("confirmation").fetch("minimum_parent_gain"),
      decision.fetch("control_best_macro") + protocol.fetch("confirmation").fetch("minimum_control_gain")].max
    failures << "Second seed did not confirm retained development gain" unless confirmation && confirmation.fetch("key").first >= minimum
    reference = evaluations.fetch("release")
    candidate = evaluations.fetch("candidate-1337")
    gates = protocol.fetch("publication")
    %w[en-US zh-CN].each do |language|
      cell = candidate.fetch("robust_by_language").fetch(language)
      gain = candidate.fetch("factual_by_language").fetch(language).fetch("source_macro_accuracy") - reference.fetch("factual_by_language").fetch(language).fetch("source_macro_accuracy")
      failures << "#{language}: factual gain below 3 points" if gain < gates.fetch("minimum_parent_gain")
      natural = candidate.fetch("natural_by_language").fetch(language).fetch("source_macro_accuracy") - reference.fetch("natural_by_language").fetch(language).fetch("source_macro_accuracy")
      failures << "#{language}: natural transfer regressed" if natural < gates.fetch("minimum_natural_gain")
      routing = candidate.fetch("routing_by_language").fetch(language).fetch("accuracy") - reference.fetch("routing_by_language").fetch(language).fetch("accuracy")
      failures << "#{language}: routing regressed beyond tolerance" if routing < -gates.fetch("maximum_routing_regression")
      { "known_accuracy" => "minimum_known_accuracy", "unknown_accuracy" => "minimum_unknown_accuracy",
        "binding_all_correct" => "minimum_binding_group_accuracy", "mixed_binding_all_correct" => "minimum_mixed_binding_group_accuracy",
        "wording_accuracy" => "minimum_wording_accuracy" }.each do |metric, rule|
        failures << "#{language}: #{metric} below declared useful-preview floor" if cell.fetch(metric) < gates.fetch(rule)
      end
    end
    failures
  end

  def publish(root, policy, result)
    destination = File.join(ROOT, "runs/decision/v0.3-factual-preview")
    raise "Preview destination exists" if File.exist?(destination)
    loaded = EasyAI::Decision::Checkpoint.load(policy.fetch("checkpoint"))
    raise "Accepted weights changed" unless loaded.fetch(:weights_fingerprint) == policy.fetch("weights_sha256")
    checkpoint = EasyAI::Decision::Checkpoint.save(destination, model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer),
      training_state: loaded.fetch(:metadata).fetch("training"), calibration: { "temperature" => policy.fetch("temperature"),
        "dataset_sha256" => sha(File.join(root, "data/calibration.jsonl")),
        "groups" => raw_rows(File.join(root, "data/calibration.jsonl")).map { |row| row.fetch("group_id") }.uniq })
    write(File.join(destination, "capabilities.json"), { "version" => "0.3", "status" => "preview", "checkpoint" => checkpoint,
      "report" => result, "scope" => "Measured Chinese/English candidate scoring; no universal semantics or business actions" })
    destination
  end

  def report(root)
    protocol = verify_opened(root)
    evaluations = JSON.parse(File.read(File.join(root, "evaluation.json")))
    policies = JSON.parse(File.read(File.join(root, "calibration.json")))
    policies.each do |_, policy|
      path = File.join(policy.fetch("checkpoint"), "weights.pt")
      raise "Evaluated weights changed" unless sha(path) == policy.fetch("weights_sha256")
    end
    failures = publication_failures(root, evaluations, protocol)
    failures << "Replay uses observed panels and cannot establish new independent acceptance" if protocol["replay_of"]
    names = evaluations.keys.reject { |name| name == "release" }
    names.each { |name| subprocess(root, "runtime", "--arm", name) }
    runtime = names.to_h { |name| [name, JSON.parse(File.read(File.join(root, "runtime-#{name}.json")))] }
    failures << "Candidate runtime checks failed" unless runtime.fetch("candidate-1337").fetch("passed")
    intervals = evaluations.keys.to_h do |name|
      rows = raw_rows(File.join(root, "predictions-#{name}.jsonl"))
      [name, DecisionFactual.bootstrap(rows)]
    end
    result = { "passed" => failures.empty?, "publication_failures" => failures, "evaluations" => evaluations,
      "group_bootstrap" => intervals, "training_exposure" => ARMS.to_h { |arm| [arm, exposure(root, arm)] },
      "runtime" => runtime, "protocol_sha256" => sha(File.join(root, "protocol.json")),
      "shared_probe_scope" => "The fixed fitting probe belongs to candidate training; its generated worlds are not control training examples. Control probe NLL is transfer, not training fit.",
      "scope" => protocol.fetch("scope"), "comparison" => "Same parent/config/update count/replay allocation; data, objective exposure and tokens differ" }
    result["reference_baselines"] = position_references(root)
    write(File.join(root, "reference-baselines.json"), result.fetch("reference_baselines"))
    result["development_summary"] = ARMS.to_h do |arm|
      trace = JSON.parse(File.read(File.join(root, "#{arm}-1337/validation.json")))
      best = trace.max_by { |row| row.fetch("factual_macro_accuracy") }
      [arm, { "eligible_checkpoints" => trace.count { |row| row.fetch("eligible") }, "best_macro_step" => best.fetch("step"),
        "best_macro" => best.fetch("factual_macro_accuracy"), "best_macro_slices" => best.fetch("robust_by_language"),
        "last_slices" => trace.last.fetch("robust_by_language"), "last_routing" => trace.last.fetch("routing_by_language"),
        "gpu_process_mib" => trace.filter_map { |row| row["gpu_process_mib"] } }]
    end
    result["preview"] = publish(root, policies.fetch("candidate-1337"), result) if failures.empty?
    write(File.join(root, "report.json"), result)
    plot(root)
    puts JSON.pretty_generate(result.slice("passed", "publication_failures", "preview"))
  end

  def plot(root)
    ARMS.each do |arm|
      directory = File.join(root, "#{arm}-1337")
      history = raw_rows(File.join(directory, "choice/training.jsonl"))
      losses = history.map { |row| row.fetch("train_loss") }
      values = history.each_with_index.map do |row, index|
        window = losses[[index - 49, 0].max..index]
        [row.fetch("step"), row.fetch("train_loss"), window.sum.fdiv(window.size), row.fetch("gradient_norm_preclip")].join("\t")
      end
      File.write(File.join(directory, "training.tsv"), values.join("\n") + "\n")
      validation = JSON.parse(File.read(File.join(directory, "validation.json")))
      values = validation.map do |row|
        cells = row.fetch("robust_by_language").values
        [row.fetch("step"), row.fetch("factual_nll"), row.fetch("factual_macro_accuracy"), row.fetch("fixed_train_probe").fetch("factual_nll"),
          cells.sum { |cell| cell.fetch("binding_all_correct") }.fdiv(2), cells.sum { |cell| cell.fetch("unknown_accuracy") }.fdiv(2),
          cells.all? { |cell| cell.key?("mixed_binding_all_correct") } ? cells.sum { |cell| cell.fetch("mixed_binding_all_correct") }.fdiv(2) : "NaN"].join("\t")
      end
      File.write(File.join(directory, "validation.tsv"), values.join("\n") + "\n")
    end
    script = <<~GNUPLOT
      set terminal pngcairo size 1400,950 enhanced font 'Sans,11'
      set output 'comparison.png'
      set multiplot layout 2,3 rowsfirst title 'Decision v0.3: common v0.1 parent, CE supervision packages (development only)'
      set xlabel 'Optimizer update'
      set grid
      set key top right
      set title 'Sampled CE: raw and 50-update moving average'
      plot 'control-1337/training.tsv' using 1:2 with lines lc rgb '#bfdbfe' title 'Control raw', \
        'candidate-1337/training.tsv' using 1:2 with lines lc rgb '#fed7aa' title 'Candidate raw', \
        'control-1337/training.tsv' using 1:3 with lines lw 2 lc rgb '#2563eb' title 'Control mean', \
        'candidate-1337/training.tsv' using 1:3 with lines lw 2 lc rgb '#ea580c' title 'Candidate mean'
      set title 'Shared candidate-training probe NLL (control: transfer)'
      plot 'control-1337/validation.tsv' using 1:4 with linespoints lc rgb '#2563eb' title 'Control', 'candidate-1337/validation.tsv' using 1:4 with linespoints lc rgb '#ea580c' title 'Candidate'
      set title 'Held-out factual NLL'
      plot 'control-1337/validation.tsv' using 1:2 with linespoints lc rgb '#2563eb' title 'Control', 'candidate-1337/validation.tsv' using 1:2 with linespoints lc rgb '#ea580c' title 'Candidate'
      set yrange [0:1]
      set title 'Held-out factual macro accuracy'
      plot 'control-1337/validation.tsv' using 1:3 with linespoints lc rgb '#2563eb' title 'Control', 'candidate-1337/validation.tsv' using 1:3 with linespoints lc rgb '#ea580c' title 'Candidate'
      set title 'Held-out actor-switch groups, both correct'
      plot 'control-1337/validation.tsv' using 1:5 with linespoints lc rgb '#2563eb' title 'Control overall', \
        'candidate-1337/validation.tsv' using 1:5 with linespoints lc rgb '#ea580c' title 'Candidate overall', \
        'candidate-1337/validation.tsv' using 1:7 with linespoints dt 2 lc rgb '#ea580c' title 'Candidate mixed-truth'
      set title 'Held-out unknown accuracy'
      plot 'control-1337/validation.tsv' using 1:6 with linespoints lc rgb '#2563eb' title 'Control', 'candidate-1337/validation.tsv' using 1:6 with linespoints lc rgb '#ea580c' title 'Candidate'
      unset multiplot
    GNUPLOT
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Chart generation failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
  end
end
