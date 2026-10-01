#!/usr/bin/env ruby
require_relative "evidence_evaluation" unless defined?(EvidenceEvaluation)

module DecisionSharedBenchmarks
  REVISION = "f8ce71361165846101d02ebc83ad44e47ae44fc3".freeze
  DOWNLOADS = File.join(EvidenceExperiment::ROOT, "data/decision/downloads/shared-benchmarks")
  module_function

  SNAPSHOTS = {
    "jev-original.jsonl" => "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "jev-easy.jsonl" => "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "jev-hard.jsonl" => "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
    "typed-000.json" => "7a21ab4168eeb685fab46c21274c4d1f64472ceda424d21d1b36752c4bb0d6a2",
    "typed-001.json" => "018ebda04d5cba213730a92618064ffe57329bd71342354b6ad23eba21a4f766",
    "typed-002.json" => "d5227238cf49f83267db97c20e0b6edb82552afa94273b1ee8e6c329db5dfd0a",
    "typed-003.json" => "472683263c1d5976301fa61d8a6faef8268bb16cd22c3bd6e8ba0619ec0f3a25"
  }.freeze

  def source_url(name)
    if name.start_with?("jev-")
      tier = name.delete_prefix("jev-").delete_suffix(".jsonl")
      "https://raw.githubusercontent.com/fstandhartinger/jevbench/#{REVISION}/datasets/public/#{tier}.jsonl"
    else
      offset = Integer(name.delete_prefix("typed-").delete_suffix(".json"), 10) * 100
      "https://datasets-server.huggingface.co/rows?dataset=LocalLLaMA/typed-decisions&config=all&split=test&offset=#{offset}&length=100"
    end
  end

  def verify_snapshot(path, expected)
    raise ArgumentError, "Benchmark snapshot changed: #{path}" unless Digest::SHA256.file(path).hexdigest == expected
  end

  def fetch_snapshot(name, checksum, directory)
    path = File.join(directory, name)
    if File.exist?(path)
      verify_snapshot(path, checksum)
    else
      EasyAI::Decision::Data::Download.fetch(source_url(name), path, expected_sha256: checksum, max_mib: 20)
    end
    puts "Verified #{name}"
  end

  def download(directory = DOWNLOADS)
    SNAPSHOTS.each { |name, checksum| fetch_snapshot(name, checksum, directory) }
  end

  def prepare(root)
    output = File.join(root, "shared-benchmarks")
    raise ArgumentError, "Panel already exists" if File.exist?(output)
    SNAPSHOTS.each { |name, checksum| verify_snapshot(File.join(DOWNLOADS, name), checksum) }
    rows = %w[original easy hard].flat_map do |tier|
      File.readlines(File.join(DOWNLOADS, "jev-#{tier}.jsonl")).map do |line|
        EasyAI::Decision::Data::BenchmarkAdapter.jev(JSON.parse(line), tier: tier)
      end
    end
    cases = 4.times.flat_map { |i| JSON.parse(File.read(File.join(DOWNLOADS, "typed-#{format('%03d', i)}.json"))).fetch("rows") }
    raise ArgumentError, "Truncated dataset-server cells" if cases.any? { |row| row.fetch("truncated_cells", []).any? }
    typed = cases.flat_map { |row| EasyAI::Decision::Data::BenchmarkAdapter.typed(row.fetch("row")) }
    raise ArgumentError, "Wrong public benchmark counts" unless rows.size == 231 && typed.size == 2000 && cases.size == 400
    raise ArgumentError, "Duplicate decisions" unless (rows + typed).map { |row| row["id"] }.uniq.size == 2231
    FileUtils.mkdir_p(output)
    EvidenceExperiment.write_rows(File.join(output, "test.jsonl"), rows + typed)
    SemanticCoverage.write_json(File.join(output, "manifest.json"), { "revision" => REVISION, "counts" => { "JevBench" => 231, "TypedDecisions" => 2000 },
      "test_sha256" => Digest::SHA256.file(File.join(output, "test.jsonl")).hexdigest,
      "sources_sha256" => Dir.glob(File.join(DOWNLOADS, "{jev-*.jsonl,typed-???.json}")).to_h { |path| [File.expand_path(path), Digest::SHA256.file(path).hexdigest] },
      "scope" => "Evaluation only, added after the primary experiment completed at user request. All public decisions retained; input-limit failures count as wrong in full-denominator accuracy. No test training, selection or temperature refit. Criteria descriptions and label sets retained; label prefixes included in option text. State is the provided string or canonical serialized object. Our single-question interface differs from Typed Decisions' all-five-questions request. Typed gold is teacher agreement, not independently verified correctness. No official leaderboard/composite/speed comparison claimed." })
    puts "Prepared shared benchmarks: 231 + 2000 decisions"
  end

  def prediction(probs)
    probs.keys.sort.max_by { |label| probs.fetch(label) }
  end

  def valid_vector?(row)
    probs = row["probabilities"]
    return false unless probs.is_a?(Hash) && !probs.empty? && probs.key?(row.fetch("target"))
    labels = row["labels"] || row.fetch("benchmark")["reference_probabilities"]&.keys
    return false if labels && probs.keys.to_set != labels.to_set
    probs.values.all? { |p| p.is_a?(Numeric) && p.finite? && p.between?(0.0, 1.0) } && (probs.values.sum - 1.0).abs <= 0.001
  end

  def metrics(rows)
    raise ArgumentError, "Empty benchmark slice" if rows.empty?
    valid, failures = rows.partition { |row| valid_vector?(row) }
    correct = valid.to_h { |row| [row.fetch("id"), prediction(row.fetch("probabilities")) == row.fetch("target")] }
    result = { "count" => rows.size, "supported" => valid.size, "coverage" => valid.size.fdiv(rows.size),
      "correct" => correct.values.count(true), "accuracy" => correct.values.count(true).fdiv(rows.size),
      "errors" => failures.group_by { |row| row["probabilities"] ? "invalid_vector" : row["error"] || "missing_prediction" }.transform_values(&:size),
      "strict_valid_vectors" => valid.size }
    unless valid.empty?
      result["supported_accuracy"] = correct.values.count(true).fdiv(valid.size)
      result["probability_quality_supported_only"] = probability_quality(valid)
    end
    groups = rows.group_by { |row| row.fetch("group_id") }.values
    result["groups"] = { "count" => groups.size, "all_correct" => groups.count { |items| items.all? { |row| correct[row.fetch("id")] } }.fdiv(groups.size) }
    result
  end

  def probability_quality(rows)
    values = rows.map do |row|
      probs = row.fetch("probabilities")
      gold = row.fetch("benchmark")["reference_probabilities"] || probs.keys.to_h { |label| [label, label == row.fetch("target") ? 1.0 : 0.0] }
      raise ArgumentError, "Reference labels differ" unless gold.keys.to_set == probs.keys.to_set
      raise ArgumentError, "Invalid reference distribution" unless gold.values.all? { |p| p.is_a?(Numeric) && p.finite? && p.between?(0.0, 1.0) } && (gold.values.sum - 1.0).abs < 0.0001
      total = gold.values.sum
      gold = gold.transform_values { |value| value / total }
      nll = gold.sum { |label, q| -q * Math.log([probs.fetch(label), 1e-12].max) }
      kl = gold.sum { |label, q| q.zero? ? 0.0 : q * Math.log(q / [probs.fetch(label), 1e-12].max) }
      brier = gold.sum { |label, q| (probs.fetch(label) - q)**2 }
      [nll, kl, brier]
    end
    { "count" => rows.size, "reference_cross_entropy" => values.sum(&:first).fdiv(rows.size),
      "reference_kl" => values.sum { |row| row[1] }.fdiv(rows.size), "reference_brier_sum" => values.sum(&:last).fdiv(rows.size),
      "scope" => "Supported cases only; Brier is sum of squared differences to the reference vector, averaged over decisions. Typed reference vectors normalized for stored rounding. Jev reference is one-hot expected label. Failures have no invented probabilities." }
  end

  def evaluate(root, arm, seed)
    EvidenceEvaluation.assert_complete(root)
    directory = File.join(root, "shared-benchmarks")
    manifest = JSON.parse(File.read(File.join(directory, "manifest.json")))
    path = File.join(directory, "test.jsonl")
    raise ArgumentError, "Panel changed" unless Digest::SHA256.file(path).hexdigest == manifest.fetch("test_sha256")
    primary = JSON.parse(File.read(File.join(root, "evaluation-#{arm}-#{seed}.json")))
    loaded = EasyAI::Decision::Checkpoint.load(primary.fetch("checkpoint"))
    raise ArgumentError, "Weights changed" unless loaded[:weights_fingerprint] == primary.fetch("weights_sha256")
    tokenizer, model = loaded.values_at(:tokenizer, :model)
    device = EasyAI::Runtime::DevicePolicy.new(requested: "auto", budget_mib: 4096).resolve
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: device, batch_size: 8)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: model.config)
    calibrator = EasyAI::Decision::Calibrator.new(temperature: primary.fetch("temperature"))
    raw = EvidenceEvaluation.raw_rows(path)
    eligible, rejected = [], {}
    raw.each do |row|
      begin
        example = EasyAI::Decision::Data::Example.new(row)
        collator.state_tokens(example.state)
        example.options.each { |option| collator.option_tokens(example.question, option.fetch("text")) }
        eligible << example
      rescue ArgumentError => error
        raise unless error.message.include?("exceeds")
        rejected[row.fetch("id")] = "input_limit"
      end
    end
    logits = evaluator.collect(eligible)
    distributions = eligible.each_with_index.to_h do |row, i|
      [row.id, row.options.map { |option| option.fetch("id") }.zip(calibrator.probabilities(logits[i])).to_h]
    end
    decisions = raw.map do |row|
      probs = distributions[row.fetch("id")]
      predicted = probs && prediction(probs)
      row.slice("id", "group_id", "source", "target", "benchmark").merge("probabilities" => probs,
        "labels" => row.fetch("options").map { |option| option.fetch("id") }, "prediction" => predicted, "correct" => predicted == row.fetch("target"), "error" => rejected[row.fetch("id")])
    end
    result = { "arm" => arm, "seed" => seed, "device" => device, "temperature" => calibrator.temperature,
      "by_benchmark" => decisions.group_by { |row| row.fetch("source").split('/').first }.transform_values { |rows| metrics(rows) },
      "by_source" => decisions.group_by { |row| row.fetch("source") }.transform_values { |rows| metrics(rows) },
      "by_type" => decisions.group_by { |row| "#{row.fetch('source').split('/').first}/#{row.fetch('benchmark').fetch('type')}" }.transform_values { |rows| metrics(rows) } }
    SemanticCoverage.write_json(File.join(directory, "#{arm}-#{seed}.json"), result)
    EvidenceExperiment.write_rows(File.join(directory, "predictions-#{arm}-#{seed}.jsonl"), decisions)
    puts "Shared benchmark #{arm}/#{seed}: #{result['by_benchmark'].transform_values { |value| value.slice('accuracy', 'coverage') }}"
  end

  def report(root)
    directory = File.join(root, "shared-benchmarks")
    results = (%w[parent] + EvidenceExperiment::ARMS).to_h do |arm|
      [arm, EvidenceExperiment::SEEDS.map { |seed| JSON.parse(File.read(File.join(directory, "#{arm}-#{seed}.json"))) }]
    end
    means = results.transform_values do |runs|
      %w[JevBench TypedDecisions].to_h do |name|
        rows = runs.map { |run| run.fetch("by_benchmark").fetch(name) }
        raise ArgumentError, "Inconsistent seed coverage" unless rows.map { |row| row.values_at("count", "supported") }.uniq.size == 1
        [name, { "count" => rows.first["count"], "coverage" => rows.first["coverage"],
          "accuracy" => EvidenceEvaluation.average(rows.map { |row| row["accuracy"] }), "seed_accuracies" => rows.map { |row| row["accuracy"] } }]
      end
    end
    report = { "means" => means, "scope" => JSON.parse(File.read(File.join(directory, "manifest.json"))).fetch("scope") }
    SemanticCoverage.write_json(File.join(directory, "report.json"), report)
    puts JSON.pretty_generate(means)
    report
  end
end

if $PROGRAM_NAME == __FILE__
  options = { phase: "all", output: "runs/decision/evidence-v1", arm: "answer", seed: 1337 }
  OptionParser.new do |parser|
    %i[phase output arm].each { |key| parser.on("--#{key} VALUE") { |value| options[key] = value } }
    parser.on("--seed N", Integer) { |value| options[:seed] = value }
  end.parse!
  root = File.expand_path(options[:output])
  case options[:phase]
  when "download" then DecisionSharedBenchmarks.download
  when "prepare" then DecisionSharedBenchmarks.prepare(root)
  when "evaluate" then DecisionSharedBenchmarks.evaluate(root, options[:arm], options[:seed])
  when "report" then DecisionSharedBenchmarks.report(root)
  when "all"
    DecisionSharedBenchmarks.download
    DecisionSharedBenchmarks.prepare(root)
    (%w[parent] + EvidenceExperiment::ARMS).product(EvidenceExperiment::SEEDS).each do |arm, seed|
      command = [RbConfig.ruby, __FILE__, "--phase", "evaluate", "--output", root, "--arm", arm, "--seed", seed.to_s]
      raise "Shared benchmark evaluation failed" unless system(*command)
    end
    DecisionSharedBenchmarks.report(root)
  else raise ArgumentError, "Expected download, prepare, evaluate, report or all"
  end
end
