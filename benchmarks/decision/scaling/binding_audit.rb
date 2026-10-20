module DecisionScaling
  module_function

  def independent_actor_pairs(rows, predictions)
    raise ArgumentError, "Prediction count mismatch" unless rows.size == predictions.size
    selected = rows.each_index.select do |index|
      facts = rows[index].fetch("world").fetch("facts")
      facts.size == 2 && facts.map { |fact| fact.fetch("actor") }.uniq.size == 2 &&
        facts.map { |fact| fact.fetch("event") }.uniq.size == 1 && facts.map { |fact| fact.fetch("truth") }.uniq.size == 2
    end
    selected.group_by { |index| rows[index].fetch("language") }.transform_values do |indexes|
      # Reconstruct pairs from the rendered record/assertion, without contrast IDs.
      pairs = indexes.group_by { |index| [rows[index].fetch("state"), rows[index].dig("world", "assertion")] }.values
      unless pairs.all? { |pair| pair.size == 2 && pair.map { |index| rows[index].fetch("target") }.sort == %w[no yes] }
        raise ArgumentError, "Actor audit requires complete opposite-label binary pairs"
      end
      counts = pairs.map { |pair| pair.count { |index| predictions[index] == rows[index].fetch("target") } }.tally
      { "groups" => pairs.size, "rows" => indexes.size, "correct_rows" => indexes.count { |index| predictions[index] == rows[index].fetch("target") },
        "row_accuracy" => indexes.count { |index| predictions[index] == rows[index].fetch("target") }.fdiv(indexes.size),
        "zero_correct_pairs" => counts.fetch(0, 0), "one_correct_pairs" => counts.fetch(1, 0), "two_correct_pairs" => counts.fetch(2, 0),
        "all_correct" => counts.fetch(2, 0).fdiv(pairs.size),
        "same_prediction_pairs" => pairs.count { |pair| pair.map { |index| predictions[index] }.uniq.size == 1 },
        "independent_uniform_random_expected_row_accuracy" => 0.5, "independent_uniform_random_expected_all_correct" => 0.25 }
    end
  end

  def binding_audit(root, profile)
    verify(root)
    raise ArgumentError, "Unregistered audit capacity" unless %w[base large].include?(profile.to_s)
    result_path = File.join(root, "arms", "#{profile}-512-1337", "extended.json")
    result = JSON.parse(File.read(result_path))
    rows = panel(root, "novel_actor")
    EasyAI::Decision::Data::QualityAudit.integrity!(rows)
    loaded = EasyAI::Decision::Checkpoint.load(result.fetch("checkpoint"))
    _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, "cpu", gc_interval: 8)
    predictions = rows.each_index.map { |index| rows[index].fetch("options").fetch(logits[index].each_index.max_by { |option| logits[index][option] }).fetch("id") }
    independent = independent_actor_pairs(rows, predictions)
    official = DecisionJudgment.metrics(rows, logits)
    recorded = result.fetch("development").fetch("novel_actor")
    independent.each do |language, cell|
      actual = official.dig("by_language", language)
      raise "Independent group count mismatch" unless cell.fetch("groups") == actual.fetch("actor_binding_groups")
      raise "Independent correctness mismatch" unless cell.fetch("all_correct") == actual.fetch("actor_binding_all_correct")
      raise "CPU/checkpoint result differs from recorded CUDA result" unless (cell.fetch("all_correct") - recorded.dig("by_language", language, "actor_binding_all_correct")).abs < 1e-12
    end
    predictor = EasyAI::Decision::Predictor.load(result.fetch("checkpoint"), device: :cpu, candidate_chunk_size: 1)
    samples = rows.each_index.select { |index| rows[index].dig("world", "mixed") && rows[index].fetch("source").end_with?("Actor") }.group_by { |index| rows[index].fetch("language") }.values.flat_map { |items| items.first(4) }
    public_checks = samples.map do |index|
      row = rows[index]
      value = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: row.fetch("options").reverse)
      guess = value.fetch("probabilities").max_by { |_, probability| probability }.first
      raise "Public predictor/candidate mapping mismatch" unless guess == predictions[index]
      row.slice("id", "language", "state", "question", "target").merge("prediction" => guess, "probabilities" => value.fetch("probabilities"))
    end
    output = File.join(root, "binding-audit")
    FileUtils.mkdir_p(output)
    prediction_path = File.join(output, "#{profile}-predictions.jsonl")
    DecisionRobust.write_rows(prediction_path, rows.each_index.map { |index| rows[index].merge("logits" => logits[index], "prediction" => predictions[index]) })
    value = { "checkpoint" => result.fetch("checkpoint"), "weights_sha256" => sha(File.join(result.fetch("checkpoint"), "weights.pt")),
      "predictions_sha256" => sha(prediction_path), "independent_actor_pairs" => independent,
      "official_metrics" => compact_metrics(official), "public_predictor_checks" => public_checks,
      "scope" => "Observed novel-actor binary development only. CPU inference, independent grouping without contrast IDs, agreement with recorded CUDA metrics, independent gold/render integrity and public reversed-candidate ID checks. No optimizer updates." }
    write(File.join(output, "#{profile}.json"), value)
    puts JSON.pretty_generate(independent)
    value
  end

  def binding_input_audit(root)
    protocol = verify(root)
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    config = EasyAI::Decision::Config.new(protocol.fetch("config_by_profile").fetch("base")).with(input: { truncation: "error" })
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: config)
    names = %w[train-512-binary validation validation-novel_actor]
    values = names.to_h do |name|
      rows = DecisionRelational.rows(File.join(root, "data", "#{name}.jsonl")).select { |row| row.dig("world", "profile") == "binary" }
      seen = {}
      rows.each do |row|
        ids = [collator.state_tokens(row.fetch("state")), row.fetch("options").map { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }]
        key = Digest::SHA256.hexdigest(JSON.generate(ids))
        raise "Opposite targets have identical actual model inputs" if seen.key?(key) && seen.fetch(key) != row.fetch("target")
        seen[key] = row.fetch("target")
      end
      [name, { "rows" => rows.size, "unique_tokenized_inputs" => seen.size, "opposite_label_collisions" => 0 }]
    end
    value = { "panels" => values, "truncations" => collator.truncated, "tokenizer_fingerprint" => protocol.fetch("tokenizer_fingerprint"),
      "scope" => "Actual own-parent NativeBpe plus experimental collator/caps; compare complete state and every question-option token sequence. Read-only, no optimizer or model inference." }
    FileUtils.mkdir_p(File.join(root, "binding-audit"))
    write(File.join(root, "binding-audit/input-pipeline.json"), value)
    puts JSON.pretty_generate(value)
    value
  end
end
