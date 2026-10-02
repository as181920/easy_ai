require_relative "natural" unless defined?(DecisionNatural)

module DecisionNaturalEvaluation
  module_function

  def assert_complete(root)
    DecisionNatural::MIXTURES.product(DecisionNatural::POSITIONS).each do |mixture, position|
      path = File.join(root, "#{mixture}-#{position}/summary.json")
      raise ArgumentError, "All pilot budgets must finish before test evaluation" unless File.file?(path)
      summary = JSON.parse(File.read(path))
      raise ArgumentError, "All pilot budgets must finish before test evaluation" unless summary.fetch("step") == DecisionNatural::STEPS
    end
  end

  def task_metrics(rows, logits, calibrator)
    rows.each_index.group_by { |i| "#{rows[i].source}/#{rows[i].language}" }.transform_values do |indexes|
      subset = indexes.map { |i| rows[i] }
      values = indexes.map { |i| logits[i] }
      metrics = EasyAI::Decision::Evaluator.metrics(values, subset.map(&:target_index), calibrator)
      labels = subset.first.options.map { |option| option.fetch("id") }
      counts = subset.group_by(&:target).transform_values(&:size)
      recalls = subset.each_index.group_by { |i| subset[i].target }.transform_values do |items|
        { "count" => items.size, "accuracy" => items.count { |i| values[i].each_index.max_by { |j| values[i][j] } == subset[i].target_index }.fdiv(items.size) }
      end
      metrics.merge("chance_accuracy" => 1.0 / labels.size, "majority_accuracy" => counts.values.max.fdiv(subset.size),
        "by_label" => recalls, "balanced_accuracy_observed_labels" => recalls.values.sum { |item| item.fetch("accuracy") }.fdiv(recalls.size),
        "unrepresented_labels" => labels - counts.keys, "independent_groups" => subset.map(&:group_id).uniq.size)
    end
  end

  def source_macro(tasks, metric)
    tasks.group_by { |name, _row| name.split('/').first }.values.map do |items|
      items.sum { |_name, row| row.fetch(metric) }.fdiv(items.size)
    end.then { |values| values.sum.fdiv(values.size) }
  end

  def evaluate(root, mixture, position)
    assert_complete(root)
    protocol = DecisionNatural.verify(root)
    directory = File.join(root, "#{mixture}-#{position}")
    if mixture == "parent"
      raise ArgumentError, "Parent reference uses its original sinusoidal positions" unless position == "sinusoidal"
      FileUtils.mkdir_p(directory)
      loaded = EasyAI::Decision::Checkpoint.load(protocol.fetch("initial"))
      raise "Initial checkpoint changed" unless loaded.fetch(:weights_fingerprint) == protocol.fetch("initial_sha256")
    else
      loaded = EasyAI::Decision::Checkpoint.load(File.join(directory, "selected"))
    end
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    evaluator = SemanticCoverageEvaluation.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), device: device, batch_size: 4)
    calibration = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/calibration.jsonl")).to_a
    calibration_logits = evaluator.collect(calibration)
    calibrator = EasyAI::Decision::Calibrator.new.fit(calibration_logits, calibration.map(&:target_index))
    result = { "mixture" => mixture, "position" => position, "device" => device, "checkpoint" => loaded.fetch(:path),
      "weights_sha256" => loaded.fetch(:weights_fingerprint), "temperature" => calibrator.temperature,
      "calibration_count" => calibration.size, "protocol_sha256" => Digest::SHA256.file(File.join(root, "protocol.json")).hexdigest }
    %w[test heldout].each do |split|
      rows = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/#{split}.jsonl")).to_a
      logits = evaluator.collect(rows)
      write_predictions(File.join(directory, "predictions-#{split}.jsonl"), rows, logits, calibrator)
      tasks = task_metrics(rows, logits, calibrator)
      result[split] = { "tasks" => tasks, "macro_source_accuracy" => source_macro(tasks, "accuracy"), "macro_source_nll" => source_macro(tasks, "nll"),
        "raw_tasks" => task_metrics(rows, logits, EasyAI::Decision::Calibrator.new) }
    end
    test = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/test.jsonl")).to_a
    # Perturbations only on fresh QA/NLI; do not imply new gold labels.
    relation_rows = test.select { |row| %w[BoolQ DuReader-YesNo OCNLI].include?(row.source) }
    result["state_question_sensitivity"] = evaluator.diagnostics(relation_rows)
    # Observed controlled panel is a regression diagnostic, never a fresh target.
    binding = File.join(EvidenceExperiment::ROOT, "runs/decision/evidence-v1/data/test.jsonl")
    raise "Observed binding panel changed" unless Digest::SHA256.file(binding).hexdigest == protocol.fetch("sources_sha256").fetch(binding)
    result["binding_regression"] = EvidenceEvaluation.controlled(loaded.fetch(:model), loaded.fetch(:tokenizer), binding, device, calibrator)
    checkpoint = EasyAI::Decision::Checkpoint.save(File.join(directory, "calibrated"), model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer),
      calibration: { "temperature" => calibrator.temperature, "rows" => calibration.size, "scope" => protocol.fetch("scope") })
    result["calibrated_checkpoint"] = checkpoint
    SemanticCoverage.write_json(File.join(root, "evaluation-#{mixture}-#{position}.json"), result)
    puts "Evaluated #{mixture}/#{position}: main=#{result.dig('test', 'macro_source_accuracy').round(4)} heldout=#{result.dig('heldout', 'macro_source_accuracy').round(4)}"
  end

  def write_predictions(path, rows, logits, calibrator)
    File.open(path, "w") do |file|
      rows.each_with_index do |row, i|
        predicted = row.options.fetch(logits[i].each_index.max_by { |j| logits[i][j] }).fetch("id")
        file.puts(JSON.generate("id" => row.id, "group_id" => row.group_id, "source" => row.source, "language" => row.language,
          "target" => row.target, "prediction" => predicted, "correct" => predicted == row.target,
          "candidate_ids" => row.options.map { |option| option.fetch("id") }, "logits" => logits[i], "probabilities" => calibrator.probabilities(logits[i])))
      end
    end
  end

  def report(root)
    assert_complete(root)
    DecisionNatural.verify(root)
    results = ([["parent", "sinusoidal"]] + DecisionNatural::MIXTURES.product(DecisionNatural::POSITIONS)).to_h do |mixture, position|
      ["#{mixture}-#{position}", JSON.parse(File.read(File.join(root, "evaluation-#{mixture}-#{position}.json")))]
    end
    comparisons = DecisionNatural::POSITIONS.to_h do |position|
      control, broad = %w[control broad].map { |mixture| results.fetch("#{mixture}-#{position}") }
      deltas = broad.dig("test", "tasks").to_h { |name, row| [name, row.fetch("accuracy") - control.dig("test", "tasks", name, "accuracy")] }
      [position, { "macro_gain" => broad.dig("test", "macro_source_accuracy") - control.dig("test", "macro_source_accuracy"),
        "heldout_gain" => broad.dig("heldout", "macro_source_accuracy") - control.dig("heldout", "macro_source_accuracy"), "task_cell_deltas" => deltas,
        "scope" => "One seed; no robustness or production gate. Trained-task improvement differs from withheld Emotion transfer." }]
    end
    table = "Condition                 Main macro  Emotion  Binding all correct\n"
    results.each do |name, result|
      table << format("%-26s %8.2f %8.2f %20.2f\n", name, result.dig("test", "macro_source_accuracy") * 100,
        result.dig("heldout", "macro_source_accuracy") * 100, result.dig("binding_regression", "groups", "binding", "all_correct") * 100)
    end
    File.write(File.join(root, "summary.txt"), table)
    SemanticCoverage.write_json(File.join(root, "report.json"), { "results" => results, "comparisons" => comparisons,
      "training_audit" => training_audit(root),
      "heldout_choice_distribution" => heldout_choice_distribution(root, results),
      "scope" => "Single-seed generalization pilot. No test-based selection and no checkpoint promotion." })
    plot(root, results)
    plot_losses(root)
    puts table
  end

  def heldout_choice_distribution(root, results)
    rows = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/heldout.jsonl")).to_a
    results.to_h do |name, _result|
      saved = EvidenceEvaluation.raw_rows(File.join(root, name, "predictions-heldout.jsonl"))
      raise "Prediction order changed" unless saved.map { |row| row.fetch("id") } == rows.map(&:id)
      [name, choice_distribution(rows, saved.map { |row| row.fetch("logits") })]
    end
  end

  def choice_distribution(rows, logits)
    raise "Expected one fixed candidate vocabulary" unless rows.map(&:options).uniq.size == 1
    targets = rows.map(&:target).tally
    predictions = rows.each_index.map { |i| rows[i].options.fetch(logits[i].each_index.max_by { |j| logits[i][j] }).fetch("id") }.tally
    dominant, count = predictions.max_by { |_label, number| number }
    { "target_counts" => targets, "prediction_counts" => predictions, "dominant_id" => dominant,
      "dominant_text" => rows.first.options.find { |option| option.fetch("id") == dominant }.fetch("text"),
      "dominant_prediction_fraction" => count.fdiv(rows.size), "dominant_target_fraction" => targets.fetch(dominant, 0).fdiv(rows.size),
      "scope" => "Post-hoc failure diagnostic from stored logits and fixed candidate definitions. Concentration suggests an option-prior hypothesis; it does not prove state independence." }
  end

  def training_audit(root)
    coverage = {}
    tokenizer = EasyAI::Tokenizers::Registry.load(File.join(root, "data/tokenizer.json"))
    runs = DecisionNatural::MIXTURES.product(DecisionNatural::POSITIONS).to_h do |mixture, position|
      name = "#{mixture}-#{position}"
      directory = File.join(root, name)
      summary = JSON.parse(File.read(File.join(directory, "summary.json")))
      traces = EvidenceEvaluation.raw_rows(File.join(directory, "choice/training.jsonl"))
      memory = EvidenceEvaluation.raw_rows(File.join(directory, "choice/metrics.jsonl"))
      coverage[name] = JSON.parse(File.read(File.join(directory, "coverage.json"))).fetch("row_visits")
      exposure = reconstructed_exposure(File.join(root, "data/#{mixture}.jsonl"), coverage.fetch(name), tokenizer)
      tokens = exposure.fetch("input_tokens")
      selected = JSON.parse(File.read(File.join(directory, "selected/metadata.json"))).fetch("training")
      selected_exposure = reconstructed_exposure(File.join(root, "data/#{mixture}.jsonl"), selected.fetch("coverage").fetch("row_visits"), tokenizer)
      [name, summary.merge("trace_rows" => traces.size, "trace_devices" => traces.map { |row| row.fetch("device") }.uniq,
        "reconstructed_input_tokens" => tokens, "input_token_counter_matches" => tokens == summary.fetch("input_tokens"),
        "exposure_by_source_language" => exposure.fetch("by_source_language"),
        "selected_checkpoint_exposure" => selected.slice("step", "examples_seen").merge(selected_exposure),
        "validation_memory" => memory.map { |row| row.slice("step", "gpu_process_mib_before_validation", "gpu_process_mib_after_validation") })]
    end
    { "runs" => runs, "same_row_visits_within_mixture" => DecisionNatural::MIXTURES.to_h do |mixture|
      [mixture, coverage.fetch("#{mixture}-sinusoidal") == coverage.fetch("#{mixture}-rotary")]
    end }
  end

  def reconstructed_tokens(path, visits, tokenizer)
    reconstructed_exposure(path, visits, tokenizer).fetch("input_tokens")
  end

  def reconstructed_exposure(path, visits, tokenizer)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: DecisionNatural.config("sinusoidal"))
    dataset = EasyAI::Decision::Data::Dataset.new(path)
    raise "Coverage row count mismatch" unless visits.size == dataset.size
    cells = Hash.new { |hash, key| hash[key] = { "rows" => 0, "unique_rows_seen" => 0, "examples_seen" => 0, "input_tokens" => 0 } }
    dataset.each_with_index do |row, i|
      cell = cells["#{row.source}/#{row.language}"]
      cell["rows"] += 1
      next if visits[i].zero?
      cell["unique_rows_seen"] += 1
      cell["examples_seen"] += visits[i]
      cell["input_tokens"] += visits[i] * (collator.state_tokens(row.state).size + row.options.sum { |option| collator.option_tokens(row.question, option.fetch("text")).size })
    end
    { "input_tokens" => cells.values.sum { |cell| cell.fetch("input_tokens") }, "by_source_language" => cells }
  end

  def plot(root, results)
    File.write(File.join(root, "comparison.tsv"), results.map do |name, result|
      label = name.sub("parent-sinusoidal", "parent").sub("control", "old").sub("sinusoidal", "sin").sub("rotary", "RoPE")
      [label, result.dig("test", "macro_source_accuracy"), result.dig("heldout", "macro_source_accuracy"),
        result.dig("binding_regression", "groups", "binding", "all_correct")].join("\t")
    end.join("\n") + "\n")
    script = "set terminal pngcairo size 1200,700\nset output 'comparison.png'\nset title 'Natural-task pilot: seed 1337; unchanged parent and validation-selected fits'\nset style data histograms\nset style histogram clustered gap 1\nset style fill solid .8 border -1\nset yrange [0:1]\nset ylabel 'Fraction correct'\nset key outside\nplot 'comparison.tsv' using 2:xtic(1) title 'Main source macro', '' using 3 title 'Withheld Emotion', '' using 4 title 'Observed binding regression'\n"
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Plot failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
  end

  def plot_losses(root)
    script = "set terminal pngcairo size 1400,950\nset output 'loss-comparison.png'\nset multiplot layout 2,2 title 'Natural-task pilot: training trailing mean (up to 25 updates); validation unsmoothed'\nset xlabel 'Update'\nset ylabel 'Cross-entropy'\nset yrange [0:1.8]\nset key bottom left\n"
    DecisionNatural::MIXTURES.product(DecisionNatural::POSITIONS).each do |mixture, position|
      name = "#{mixture}-#{position}"
      directory = File.join(root, name, "choice")
      train = EvidenceEvaluation.raw_rows(File.join(directory, "training.jsonl"))
      validation = EvidenceEvaluation.raw_rows(File.join(directory, "metrics.jsonl"))
      File.write(File.join(root, "loss-#{name}.tsv"), train.each_with_index.map do |row, i|
        window = train[[i - 24, 0].max..i]
        [row.fetch("step"), row.fetch("train_loss"), window.sum { |item| item.fetch("train_loss") }.fdiv(window.size)].join("\t")
      end.join("\n") + "\n")
      File.write(File.join(root, "validation-#{name}.tsv"), validation.map { |row| row.values_at("step", "validation_loss").join("\t") }.join("\n") + "\n")
      script << "set title '#{name}'\nplot 'loss-#{name}.tsv' using 1:3 with lines title 'Train trailing mean', 'validation-#{name}.tsv' using 1:2 with linespoints title 'Common validation'\n"
    end
    script << "unset multiplot\n"
    File.write(File.join(root, "loss-comparison.gnuplot"), script)
    raise "Loss plot failed" unless system("gnuplot", "loss-comparison.gnuplot", chdir: root)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/natural-v1-pilot", mixture: "control", position: "sinusoidal" }
  OptionParser.new do |parser|
    %i[phase output mixture position].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "evaluate" then DecisionNaturalEvaluation.evaluate(root, options[:mixture], options[:position])
  when "report" then DecisionNaturalEvaluation.report(root)
  when "all"
    ([["parent", "sinusoidal"]] + DecisionNatural::MIXTURES.product(DecisionNatural::POSITIONS)).each do |mixture, position|
      raise "Evaluation failed" unless system(RbConfig.ruby, __FILE__, "--phase", "evaluate", "--output", root, "--mixture", mixture, "--position", position)
    end
    DecisionNaturalEvaluation.report(root)
  else raise ArgumentError, "Unknown phase"
  end
end
