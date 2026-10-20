module DecisionEventGeneralization
  module_function

  def identities(root)
    values = [1337, 2027].to_h { |seed| ["baseline-#{seed}", { "path" => parent(seed), "weights_sha256" => sha(File.join(parent(seed), "weights.pt")) }] }
    Dir[File.join(root, "arms/*/result.json")].each do |file|
      value = sealed(file)
      values[File.basename(File.dirname(file))] = { "path" => value.fetch("checkpoint"), "weights_sha256" => value.fetch("weights_sha256") }
    end
    values
  end

  def audit_inputs(root)
    protocol = verify(root)
    path = File.join(root, "isolation-correction/input-pipeline-audit.json")
    return sealed(path) if File.exist?(path)
    loaded = EasyAI::Decision::Checkpoint.load(parent(1337))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config)
    seen = {}
    counts = protocol.fetch("files_sha256").to_h do |file, _|
      count = 0
      File.foreach(File.join(root, "data", file)) do |line|
        row = JSON.parse(line)
        ids = [collator.state_tokens(row.fetch("state")), row.fetch("options").map { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }]
        key = Digest::SHA256.hexdigest(JSON.generate(ids))
        raise "Opposite labels share tokenized inputs" if seen.key?(key) && seen.fetch(key) != row.fetch("target")
        seen[key] = row.fetch("target")
        count += 1
      end
      [file, count]
    end
    other = EasyAI::Decision::Checkpoint.load(parent(2027))
    raise "Parent tokenizer mismatch" unless loaded.fetch(:tokenizer).fingerprint == other.fetch(:tokenizer).fingerprint
    seal(path, { "rows_by_file" => counts, "rows" => counts.values.sum, "unique_tokenized_inputs" => seen.size, "opposite_label_collisions" => 0,
      "truncations" => collator.truncated, "tokenizer_fingerprint" => loaded.fetch(:tokenizer).fingerprint,
      "scope" => "Frozen actual tokenizer/collator and caps, including unopened final data. Data integrity only, no model inference or optimization." })
  end

  def crossed_row(row, split: "lockbox")
    forms = Catalog::FORMS.fetch(split.to_s)
    current = forms.index(row.fetch("world").fetch("form"))
    raise "Diagnostic requires a held template" unless current
    Catalog.render(row, form: forms.fetch((current + 1) % forms.size))
  end

  def crossed_templates(root)
    verify(root)
    opened = sealed(File.join(root, "lockbox-opened.json"))
    raise "Final choices changed" unless opened.fetch("choices_sha256") == sha(File.join(root, "final-choices.json"))
    sealed(File.join(root, "final-choices.json")).fetch("checkpoints").each do |identity, checkpoint|
      path = File.join(root, "crossed-template-diagnostic", "#{identity}.json")
      next if File.exist?(path) && sealed(path)
      raise "Diagnostic weights changed" unless sha(File.join(checkpoint.fetch("path"), "weights.pt")) == checkpoint.fetch("weights_sha256")
      loaded = EasyAI::Decision::Checkpoint.load(checkpoint.fetch("path"))
      views = %w[expressions composed].to_h do |view|
        rows = raw(root, "lockbox-#{view}").map { |row| crossed_row(row) }
        rows.each { |row| Corpus.gold!(row) }
        _, logits = DecisionRobust.collect(loaded.fetch(:model), loaded.fetch(:tokenizer), rows, "cuda", gc_interval: 8)
        raise "Crossed diagnostic fell back" unless loaded.fetch(:model).parameters.all? { |parameter| parameter.device.to_s.start_with?("cuda") }
        guesses = rows.each_index.map { |index| rows[index].fetch("options")[logits[index].each_index.max_by { |option| logits[index][option] }].fetch("id") }
        metrics = DecisionJudgment.metrics(rows, logits)
        independent = %w[actor event].to_h { |axis| [axis, independent_pairs(rows, guesses, axis: axis)] }
        independent.each do |axis, languages|
          languages.each { |language, cell| raise "Crossed pair count mismatch" unless cell.fetch("all_correct") == metrics.dig("by_language", language, "#{axis}_binding_all_correct") }
        end
        file = File.join(root, "crossed-template-diagnostic", "#{identity}-#{view}.jsonl")
        FileUtils.mkdir_p(File.dirname(file))
        DecisionRobust.write_rows(file, rows.each_index.map { |index| rows[index].merge("prediction" => guesses[index], "logits" => logits[index]) })
        puts "Crossed held template #{identity}/#{view}: #{metrics.fetch('worst_language_balanced').round(4)}"
        $stdout.flush
        [view, { "metrics" => DecisionScaling.compact_metrics(metrics), "independent_pairs" => independent, "predictions_sha256" => sha(file) }]
      end
      seal(path, { "weights_sha256" => checkpoint.fetch("weights_sha256"), "views" => views,
        "scope" => "Post-selection paired-template diagnostic registered before final opening. Swaps the two already reserved templates on the same final expression/composed world families; no new independent worlds, no model selection or optimization." })
      loaded.fetch(:model).to("cpu")
      loaded = nil
      GC.start
    end
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
      rows = raw(root, name).select { |row| row.fetch("group_id").split(":").last.to_i < 4 }
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
        .group_by { |row| [row.fetch("language"), row.fetch("source")] }.values.flat_map { |group| group.first(2) }
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
      "scope" => "Fixed first four complete worlds per language/view: 256 rows, including both axes, truth assignments, record orders and claim polarities. CPU/CUDA audit is a subset, not full-panel equivalence. Full GPU predictions have independent pair reconstruction." })
  end
end
