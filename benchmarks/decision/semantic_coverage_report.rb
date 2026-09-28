require "open3"
require "cgi"
require_relative "semantic_coverage_evaluation"

module SemanticCoverageReport
  module_function

  def protocol(root)
    value = JSON.parse(File.read(File.join(root, "protocol.json")))
    value.fetch("files_sha256").each do |file, expected|
      raise ArgumentError, "Experiment input changed: #{file}" unless Digest::SHA256.file(File.join(root, "data", file)).hexdigest == expected
    end
    value.fetch("arms").product(value.fetch("seeds")).each do |arm, seed|
      summary = JSON.parse(File.read(File.join(root, "#{arm}-#{seed}/summary.json")))
      unless summary.fetch("budgets").map { |row| row.fetch("budget") } == value.fetch("budgets")
        raise ArgumentError, "Finish all predefined runs before opening the challenge"
      end
    end
    value
  end

  def evaluate(root, arm, seed, budget, device: "auto")
    contract = protocol(root)
    unless contract.fetch("arms").include?(arm) && contract.fetch("seeds").include?(seed) && contract.fetch("budgets").include?(budget)
      raise ArgumentError, "Unknown evaluation run"
    end
    directory = File.join(root, "#{arm}-#{seed}")
    checkpoint = File.join(directory, "selected-#{budget}")
    loaded = EasyAI::Decision::Checkpoint.load(checkpoint)
    expected_config = EasyAI::Decision::Config.new(contract.fetch("config")).with(training: { seed: seed }).to_h
    unless loaded[:model].config.to_h == expected_config && loaded[:metadata].dig("training", "step") <= budget
      raise ArgumentError, "Checkpoint configuration or selection budget changed"
    end
    expected = contract.fetch("files_sha256").fetch(arm == "baseline" ? "train.jsonl" : "expanded.jsonl")
    raise ArgumentError, "Checkpoint trained on different inputs" unless loaded[:metadata].dig("training", "datasets", "train") == expected
    tokenizer_sha256 = Digest::SHA256.file(File.join(checkpoint, "tokenizer.json")).hexdigest
    raise ArgumentError, "Checkpoint tokenizer changed" unless tokenizer_sha256 == contract.fetch("tokenizer_sha256")
    inputs = %w[train validation challenge].to_h do |name|
      [name, EasyAI::Decision::Data::Dataset.new(File.join(root, "data/#{name}.jsonl"))]
    end
    signature = { "version" => 1, "weights_sha256" => loaded[:weights_fingerprint], "data_sha256" => inputs.transform_values(&:fingerprint),
      "tokenizer_sha256" => tokenizer_sha256,
      "reporter_sha256" => Digest::SHA256.file(__FILE__).hexdigest,
      "evaluator_sha256" => Digest::SHA256.file(File.join(__dir__, "semantic_coverage_evaluation.rb")).hexdigest }
    output = File.join(directory, "evaluation-#{budget}.json")
    if File.exist?(output)
      cached = JSON.parse(File.read(output))
      raise ArgumentError, "Evaluation contract changed" unless cached.fetch("contract") == signature
      return cached
    end
    policy = EasyAI::Runtime::DevicePolicy.new(requested: device, budget_mib: loaded[:model].config[:runtime]["gpu_memory_budget_mib"])
    evaluator = SemanticCoverageEvaluation.new(model: loaded[:model], tokenizer: loaded[:tokenizer], device: policy.resolve)
    training_data = EasyAI::Decision::Data::Dataset.new(File.join(root, "data", arm == "baseline" ? "train.jsonl" : "expanded.jsonl"))
    counters = loaded[:metadata].fetch("training").fetch("coverage")
    selected_coverage = EasyAI::Decision::Data::CoverageAudit.call(training_data,
      visits: counters.fetch("row_visits"), input_tokens: counters.fetch("input_tokens"))
    training_probe = inputs.fetch("train").group_by(&:source).values.flat_map do |rows|
      rows.sort_by { |row| Digest::SHA256.hexdigest("training-probe-v1:#{row.id}") }.first(100)
    end
    validation = inputs.fetch("validation").to_a
    diagnostic = validation.group_by(&:source).values.flat_map do |rows|
      groups = rows.map(&:group_id).uniq.first(50).to_set
      rows.select { |row| groups.include?(row.group_id) }
    end
    result = { "contract" => signature, "arm" => arm, "seed" => seed, "budget" => budget,
      "selected_step" => loaded[:metadata].dig("training", "step"), "calibrated" => false,
      "selected_checkpoint_batching" => loaded[:metadata].fetch("training").slice("microbatch", "gradient_accumulation"),
      "selected_checkpoint_coverage" => selected_coverage,
      "train_probe_scope" => "100 original training rows per source, fixed by ID hash; not full training accuracy",
      "train_probe" => evaluator.evaluate(training_probe), "validation" => evaluator.evaluate(validation),
      "diagnostics" => evaluator.diagnostics(diagnostic), "challenge" => evaluator.evaluate(inputs.fetch("challenge").to_a, permutations: true) }
    SemanticCoverage.write_json(output, result)
    result
  end

  def resources(directory)
    selected = EasyAI::Decision::Checkpoint.resolve(File.join(directory, "choice"))
    state = JSON.parse(File.read(File.join(selected, "metadata.json"))).fetch("training")
    measurements = File.readlines(File.join(directory, "choice/metrics.jsonl")).map { |line| JSON.parse(line) }
    memory = measurements.flat_map { |row| row.values_at("gpu_process_mib_before_validation", "gpu_process_mib_after_validation").compact }
    deltas = measurements.filter_map do |row|
      before, after = row.values_at("gpu_process_mib_before_validation", "gpu_process_mib_after_validation")
      after - before if before && after
    end
    { "final_batching" => state.slice("microbatch", "gradient_accumulation"),
      "gpu_capacity_recoveries" => File.foreach(File.join(directory, "train.log")).count { |line| line.include?("GPU capacity failure") },
      "validation_devices" => measurements.map { |row| row.fetch("device") }.uniq,
      "sampled_process_memory_mib_range" => memory.minmax,
      "validation_memory_change_mib_range" => deltas.minmax,
      "scope" => "Memory sampled around validation, not a continuous peak measurement. Batch recovery can change dropout and numerical trajectories despite preserving effective batch size." }
  end

  def comparison(before, after, gate)
    seeds = before.keys.sort
    per_seed = seeds.map do |seed|
      { "seed" => seed, "macro_accuracy_gain" => after.fetch(seed).fetch("macro_source_accuracy") - before.fetch(seed).fetch("macro_source_accuracy"),
        "nll_change" => after.fetch(seed).fetch("nll") - before.fetch(seed).fetch("nll") }
    end
    source_changes = before.values.first.fetch("by_source").keys.to_h do |source|
      [source, seeds.sum { |seed| after.fetch(seed).dig("by_source", source, "accuracy") - before.fetch(seed).dig("by_source", source, "accuracy") }.fdiv(seeds.size)]
    end
    mean = per_seed.sum { |row| row.fetch("macro_accuracy_gain") }.fdiv(seeds.size)
    improved = per_seed.count { |row| row.fetch("macro_accuracy_gain") > 0 }
    { "mean_macro_accuracy_gain" => mean, "mean_source_accuracy_changes" => source_changes, "improved_seeds" => improved,
      "per_seed" => per_seed, "passed" => mean >= gate.fetch("mean_macro_accuracy_gain") &&
        source_changes.values.min >= -gate.fetch("maximum_mean_source_regression") && improved >= gate.fetch("minimum_improved_seeds") }
  end

  def write(root)
    contract = protocol(root)
    results = contract.fetch("arms").product(contract.fetch("seeds"), contract.fetch("budgets")).map do |arm, seed, budget|
      JSON.parse(File.read(File.join(root, "#{arm}-#{seed}/evaluation-#{budget}.json")))
    end
    groups = results.group_by { |row| [row.fetch("arm"), row.fetch("budget")] }
    grouped_metrics = groups.transform_values { |rows| rows.to_h { |row| [row.fetch("seed"), row.fetch("challenge")] } }
    gate = contract.fetch("gate")
    report = { "protocol" => contract, "scope" => "All predefined seeds; selection uses validation NLL. Challenge examined after training, without further tuning. Gate is an experiment decision rule, not statistical significance or general semantic reliability.",
      "resources" => contract.fetch("arms").product(contract.fetch("seeds")).to_h do |arm, seed|
        ["#{arm}-#{seed}", resources(File.join(root, "#{arm}-#{seed}"))]
      end,
      "longer_training" => comparison(grouped_metrics.fetch(["baseline", 1000]), grouped_metrics.fetch(["baseline", 4000]), gate),
      "wording_expansion" => comparison(grouped_metrics.fetch(["baseline", 4000]), grouped_metrics.fetch(["expanded", 4000]), gate),
      "results" => results.map { |row| row.except("diagnostics").merge("challenge" => row.fetch("challenge").except("predictions")) } }
    SemanticCoverage.write_json(File.join(root, "comparison.json"), report)
    lines = ["arm\tbudget\tseed\tmacro_accuracy\tnll\tselected_step"] + results.map do |row|
      [row.fetch("arm"), row.fetch("budget"), row.fetch("seed"), row.dig("challenge", "macro_source_accuracy"),
        row.dig("challenge", "nll"), row.fetch("selected_step")].join("\t")
    end
    File.write(File.join(root, "comparison.tsv"), lines.join("\n") + "\n")
    plot(root, groups)
    rows = results.map do |row|
      "<tr><td>#{row['arm']}</td><td>#{row['budget']}</td><td>#{row['seed']}</td><td>#{row['selected_step']}</td>" \
        "<td>#{format('%.2f%%', 100 * row.dig('challenge', 'macro_source_accuracy'))}</td><td>#{format('%.4f', row.dig('challenge', 'nll'))}</td>" \
        "<td><a href=\"#{row['arm']}-#{row['seed']}/report/index.html\">Training curves</a></td></tr>"
    end
    File.write(File.join(root, "index.html"), <<~HTML)
      <!doctype html><html lang="en"><meta charset="utf-8"><title>Gold-supervised coverage experiment</title>
      <style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}img{max-width:100%}pre{white-space:pre-wrap}</style>
      <h1>Gold-supervised coverage experiment</h1><p>From-scratch student; no teacher or pretrained LLM. Fixed 1,000 / 4,000-update budgets, three seeds. Wording variants preserve source gold labels and groups.</p>
      <img src="comparison.png" alt="Challenge accuracy by budget and seed">
      <table><tr><th>Data</th><th>Budget</th><th>Seed</th><th>Selected step</th><th>Challenge macro accuracy</th><th>NLL</th><th>Report</th></tr>#{rows.join}</table>
      <h2>Predeclared decisions</h2><pre>#{CGI.escapeHTML(JSON.pretty_generate(report.slice('longer_training', 'wording_expansion')))}</pre>
      <h2>Execution conditions</h2><pre>#{CGI.escapeHTML(JSON.pretty_generate(report.fetch('resources')))}</pre>
      <p>Development selection and final challenge are separate. This is a bounded experiment, not a claim of reliable multilingual semantics.</p>
      <p><a href="comparison.json">Complete metrics</a> · <a href="comparison.tsv">TSV</a> · <a href="protocol.json">Frozen protocol</a></p></html>
    HTML
    report.except("results", "protocol")
  end

  def plot(root, groups)
    points = groups.sort_by { |(arm, budget), _rows| [arm, budget] }.each_with_index.map do |((arm, budget), rows), index|
      values = rows.sort_by { |row| row.fetch("seed") }.map { |row| 100 * row.dig("challenge", "macro_source_accuracy") }
      [index, "#{arm}-#{budget}", values.sum / values.size, *values].join("\t")
    end
    File.write(File.join(root, "plot.tsv"), points.join("\n") + "\n")
    script = <<~GNUPLOT
      set terminal pngcairo size 1100,650
      set output 'comparison.png'
      set title 'Fresh public-development challenge: all three seeds'
      set ylabel 'Source-macro accuracy (%)'
      set yrange [0:100]
      set grid ytics
      set key outside top center horizontal
      set boxwidth 0.6
      set style fill solid 0.25
      plot 'plot.tsv' using 1:3:xtic(2) with boxes title 'Mean', \
        '' using 1:4 with points pt 7 ps 1.3 title 'Seed 1337', \
        '' using 1:5 with points pt 5 ps 1.3 title 'Seed 2027', \
        '' using 1:6 with points pt 9 ps 1.3 title 'Seed 3407'
      set terminal svg size 1100,650
      set output 'comparison.svg'
      replot
    GNUPLOT
    File.write(File.join(root, "plot.gnuplot"), script)
    _stdout, error, status = Open3.capture3("gnuplot", "plot.gnuplot", chdir: root)
    raise "Chart failed: #{error}" unless status.success?
  end
end
