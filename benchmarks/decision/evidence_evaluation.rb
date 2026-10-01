require_relative "evidence" unless defined?(EvidenceExperiment)
require_relative "semantic_coverage_evaluation"

# Offline ground-truth bookkeeping; model forward sees text and sentence boundaries only.
module EvidenceEvaluation
  module_function

  def raw_rows(path)
    File.readlines(path).map { |line| JSON.parse(line) }
  end

  def collect(model, tokenizer, rows, device)
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: device)
    evaluator.collect(rows)
  end

  def controlled(model, tokenizer, path, device, calibrator)
    raw = raw_rows(path)
    rows = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    logits = collect(model, tokenizer, rows, device)
    correct = rows.each_index.map { |i| logits[i].each_index.max_by { |j| logits[i][j] } == rows[i].target_index }
    result = EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index))
    result["calibrated"] = EasyAI::Decision::Evaluator.metrics(logits, rows.map(&:target_index), calibrator)
    result["by_language"] = rows.each_index.group_by { |i| rows[i].language }.transform_values do |indexes|
      EasyAI::Decision::Evaluator.metrics(indexes.map { |i| logits[i] }, indexes.map { |i| rows[i].target_index }, calibrator)
    end
    result["by_condition"] = %w[assertion order distractor].to_h do |field|
      [field, raw.each_index.group_by { |i| raw[i].fetch("world").fetch(field).to_s }.transform_values do |indexes|
        { "count" => indexes.size, "accuracy" => indexes.count { |i| correct[i] }.fdiv(indexes.size) }
      end]
    end
    result["by_condition"]["mixed_truth"] = raw.each_index.group_by { |i| raw[i].fetch("world").fetch("facts").uniq.size == 2 }.transform_values do |indexes|
      { "count" => indexes.size, "accuracy" => indexes.count { |i| correct[i] }.fdiv(indexes.size) }
    end
    checks = raw.first.fetch("world").fetch("checks").keys
    result["groups"] = checks.to_h do |name|
      groups = raw.each_index.group_by { |i| raw[i].fetch("world").fetch("checks").fetch(name) }.values
      [name, { "count" => groups.size, "all_correct" => groups.count { |indexes| indexes.all? { |i| correct[i] } }.fdiv(groups.size) }]
    end
    if model.config[:model]["evidence_head"]
      collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: model.config)
      evidence = rows.each_slice(16).flat_map do |batch|
        values = evidence_logits(model, collator, batch, device)
        GC.start
        values.each_with_index.map { |scores, i| scores.each_index.max_by { |j| scores[j] } == batch[i].evidence_index }
      end
      result["evidence_accuracy"] = evidence.count(true).fdiv(evidence.size)
    end
    result["independent_families"] = rows.map(&:group_id).uniq.size
    result
  end

  def evidence_logits(model, collator, rows, device)
    Torch.no_grad { model.forward_with_evidence(collator.call(rows, device: device, with_evidence: true)).fetch(:evidence_logits).to_a }
  end

  def assert_complete(root)
    EvidenceExperiment::ARMS.product(EvidenceExperiment::SEEDS).each do |arm, seed|
      summary = JSON.parse(File.read(File.join(root, "#{arm}-#{seed}/summary.json")))
      raise ArgumentError, "All fixed training budgets must finish before test evaluation" unless summary.fetch("step") == 800
    end
  end

  def run(root, arm, seed)
    assert_complete(root)
    EvidenceExperiment.verify(root)
    path = arm == "parent" ? EvidenceExperiment.parent(seed) : File.join(root, "#{arm}-#{seed}/selected")
    loaded = EasyAI::Decision::Checkpoint.load(path)
    model, tokenizer = loaded.values_at(:model, :tokenizer)
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    model.to(device).eval
    # One global temperature, fitted on source-balanced calibration only.
    calibration = [File.join(root, "data/calibration.jsonl"), File.join(EvidenceExperiment::ROOT, "data/decision/semantic-public/calibration.jsonl")]
      .flat_map { |file| EasyAI::Decision::Data::Dataset.new(file).to_a }.group_by(&:source).values.flat_map do |rows|
        rows.sort_by { |row| Digest::SHA256.hexdigest("evidence-calibration:#{row.id}") }.first(96)
      end
    calibrator = EasyAI::Decision::Calibrator.new.fit(collect(model, tokenizer, calibration, device), calibration.map(&:target_index))
    result = { "arm" => arm, "seed" => seed, "checkpoint" => path, "temperature" => calibrator.temperature,
      "calibration_rows" => calibration.size, "device" => device, "weights_sha256" => loaded[:weights_fingerprint] }
    %w[test test-familiar].each do |split|
      result[split] = controlled(model, tokenizer, File.join(root, "data/#{split}.jsonl"), device, calibrator)
    end
    public_rows = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/public-test.jsonl")).to_a
    public_logits = collect(model, tokenizer, public_rows, device)
    result["public"] = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: device).evaluate(public_rows)
    result["public"]["calibrated"] = EasyAI::Decision::Evaluator.metrics(public_logits, public_rows.map(&:target_index), calibrator)
    if arm != "parent"
      checkpoint = EasyAI::Decision::Checkpoint.save(File.join(root, "#{arm}-#{seed}/calibrated"), model: model, tokenizer: tokenizer,
        calibration: { "temperature" => calibrator.temperature, "rows" => calibration.size, "scope" => "Source-balanced held-out calibration; no test selection" })
      result["calibrated_checkpoint"] = checkpoint
    end
    SemanticCoverage.write_json(File.join(root, "evaluation-#{arm}-#{seed}.json"), result)
    puts "Evaluated #{arm}/#{seed}: test=#{result['test']['accuracy'].round(4)}, binding=#{result['test']['groups']['binding']['all_correct'].round(4)}"
  end

  def report(root)
    results = (%w[parent] + EvidenceExperiment::ARMS).to_h do |arm|
      [arm, EvidenceExperiment::SEEDS.map { |seed| JSON.parse(File.read(File.join(root, "evaluation-#{arm}-#{seed}.json"))) }]
    end
    means = results.transform_values do |rows|
      { "accuracy" => average(rows.map { |row| row.dig("test", "accuracy") }),
        "binding_all_correct" => average(rows.map { |row| row.dig("test", "groups", "binding", "all_correct") }),
        "fact_flip_all_correct" => average(rows.map { |row| row.dig("test", "groups", "fact_flip", "all_correct") }),
        "familiar_accuracy" => average(rows.map { |row| row.dig("test-familiar", "accuracy") }),
        "public_macro_accuracy" => average(rows.map { |row| row.dig("public", "macro_source_accuracy") }),
        "test_calibrated_nll" => average(rows.map { |row| row.dig("test", "calibrated", "nll") }),
        "public_calibrated_nll" => average(rows.map { |row| row.dig("public", "calibrated", "nll") }) }
    end
    gains = EvidenceExperiment::SEEDS.each_index.map do |i|
      results["evidence"][i].dig("test", "groups", "binding", "all_correct") - results["answer"][i].dig("test", "groups", "binding", "all_correct")
    end
    language_ok = %w[zh-CN en-US].all? do |language|
      average(results["evidence"].map { |row| row.dig("test", "by_language", language, "accuracy") }) >=
        average(results["answer"].map { |row| row.dig("test", "by_language", language, "accuracy") }) - 0.03
    end
    public_ok = means["evidence"]["public_macro_accuracy"] >= means["parent"]["public_macro_accuracy"] - 0.03
    probability_ok = %w[test_calibrated_nll public_calibrated_nll].all? { |key| means["evidence"][key] <= means["answer"][key] }
    gate = average(gains) >= 0.05 && gains.count(&:positive?) >= 2 && language_ok && public_ok && probability_ok
    report = { "means" => means, "binding_gains" => gains, "auxiliary_benefit_gate" => gate,
      "gate_checks" => { "language" => language_ok, "public_preservation" => public_ok, "probabilities" => probability_ok },
      "task_readiness" => results["evidence"].all? { |row| row["test"]["by_language"].values.all? { |metrics| metrics["accuracy"] >= 0.95 } && row.dig("test", "groups", "binding", "all_correct") >= 0.90 },
      "scope" => "Generated controlled explicit facts; 32 held-out families, not broad multilingual semantic understanding. All three seeds retained. Familiar and novel tests share families." }
    if File.exist?(File.join(root, "generalization/manifest.json"))
      require_relative "generalization"
      report["generalization"] = DecisionGeneralization.report(root)
    end
    SemanticCoverage.write_json(File.join(root, "report.json"), report)
    sanity = EvidenceExperiment::ARMS.to_h do |arm|
      loaded = EasyAI::Decision::Checkpoint.load(File.join(root, "sanity-#{arm}-1337/selected"))
      loaded[:model].eval
      [arm, controlled(loaded[:model], loaded[:tokenizer], File.join(root, "data/sanity.jsonl"), "cpu", EasyAI::Decision::Calibrator.new)]
    end
    SemanticCoverage.write_json(File.join(root, "sanity-evaluation.json"), sanity)
    plot(root, results)
    plot_losses(root)
    report
  end

  def average(values)
    values.sum.fdiv(values.size)
  end

  def plot_losses(root)
    traces = EvidenceExperiment::ARMS.product(EvidenceExperiment::SEEDS).to_h do |arm, seed|
      rows = raw_rows(File.join(root, "#{arm}-#{seed}/choice/training.jsonl"))
      path = "loss-#{arm}-#{seed}.tsv"
      values = rows.each_with_index.map do |row, index|
        window = rows[[index - 24, 0].max..index]
        (row.values_at("step", "choice_loss", "evidence_loss", "train_loss") +
          %w[choice_loss evidence_loss].map { |key| average(window.map { |item| item.fetch(key, 0.0) }) }).join("\t")
      end
      File.write(File.join(root, path), values.join("\n") + "\n")
      [[arm, seed], path]
    end
    validation = EvidenceExperiment::ARMS.product(EvidenceExperiment::SEEDS).to_h do |arm, seed|
      path = "validation-#{arm}-#{seed}.tsv"
      rows = raw_rows(File.join(root, "#{arm}-#{seed}/choice/metrics.jsonl"))
      File.write(File.join(root, path), rows.map { |row| row.values_at("step", "validation_loss").join("\t") }.join("\n") + "\n")
      [[arm, seed], path]
    end
    script = "set terminal pngcairo size 1500,1000\nset output 'loss-components.png'\nset multiplot layout 2,3 title 'Training: 25-update moving means; validation: unsmoothed'\nset xlabel 'Optimizer update'\nset ylabel 'Loss'\nset key top right\n"
    [false, true].each do |auxiliary|
      EvidenceExperiment::SEEDS.each do |seed|
        series = EvidenceExperiment::ARMS.map do |arm|
          file = traces.fetch([arm, seed])
          column = auxiliary ? 6 : 5
          "'#{file}' using 1:#{column} with lines title '#{arm} train'"
        end
        unless auxiliary
          series += EvidenceExperiment::ARMS.map { |arm| "'#{validation.fetch([arm, seed])}' using 1:2 with linespoints title '#{arm} validation'" }
        end
        script << "set title 'Seed #{seed}: #{auxiliary ? "evidence CE" : "answer CE and validation"}'\nplot #{series.join(', ')}\n"
      end
    end
    script << "unset multiplot\n"
    File.write(File.join(root, "loss-components.gnuplot"), script)
    raise "Loss plot failed" unless system("gnuplot", "loss-components.gnuplot", chdir: root)
  end

  def plot(root, results)
    data = File.join(root, "comparison.tsv")
    File.write(data, "arm\taccuracy\tbinding\tpublic\n" + results.map do |arm, rows|
      [arm, average(rows.map { |row| row.dig("test", "accuracy") }),
        average(rows.map { |row| row.dig("test", "groups", "binding", "all_correct") }),
        average(rows.map { |row| row.dig("public", "macro_source_accuracy") })].join("\t")
    end.join("\n") + "\n")
    script = "set terminal pngcairo size 1100,650\nset output 'comparison.png'\nset title 'Evidence supervision: three-seed means'\nset style data histograms\nset style histogram clustered gap 1\nset style fill solid 0.8 border -1\nset yrange [0:1]\nset ylabel 'Fraction correct'\nset key outside\nplot 'comparison.tsv' using 2:xtic(1) title 'Answer accuracy', '' using 3 title 'Binding all correct', '' using 4 title 'Public source macro'\n"
    File.write(File.join(root, "comparison.gnuplot"), script)
    raise "Plot failed" unless system("gnuplot", "comparison.gnuplot", chdir: root)
  end
end
