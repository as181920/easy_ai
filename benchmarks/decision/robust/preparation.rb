module DecisionRobust
  module_function

  def prepare(root)
    raise "Output exists; preserve prior evidence" if File.exist?(root)
    # Snapshot history before creating this run, including every previously prepared panel.
    historical, _, history_files = DecisionRelease.history
    old = DecisionFactual.verify(CONTROL)
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    cfg = loaded.fetch(:model).config.with(training: { steps: STEPS, learning_rate: 0.0001, warmup_steps: 100,
      eval_every: 100, checkpoint_every: 100, early_stopping_patience: 0, seed: 1337, track_coverage: true })
    control = raw_rows(File.join(CONTROL, "data/train.jsonl"))
    raise "Control tokenizer differs from parent" unless EasyAI::Tokenizers::Registry.load(File.join(CONTROL, "data/tokenizer.json")).fingerprint == loaded.fetch(:tokenizer).fingerprint
    generated = EasyAI::Decision::Data::RobustFacts.splits
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: cfg)
    natural, length_audit = fresh_natural(historical, collator)
    replay = control.select { |row| row.fetch("source") == "MASSIVE-Scenario" }
    public_train = control.reject { |row| row.fetch("source").start_with?("Factual-") || row.fetch("source") == "MASSIVE-Scenario" }
    panels = { "control-train.jsonl" => control, "candidate-train.jsonl" => public_train + generated.fetch("train") + replay }
    %w[validation calibration test].each do |split|
      routing = split == "test" ? raw_rows("runs/decision/release-v01-corrective/data/test.jsonl") :
        raw_rows(File.join(CONTROL, "data/#{split}.jsonl")).select { |row| row.fetch("source") == "MASSIVE-Scenario" }
      panels["#{split}.jsonl"] = natural.fetch(split) + generated.fetch(split) + routing
    end
    panels["regression.jsonl"] = raw_rows(File.join(CONTROL, "data/acceptance.jsonl"))
    panels["fit.jsonl"] = fitting_rows(panels.fetch("candidate-train.jsonl"))
    audit_splits!(panels, historical, collator)
    FileUtils.mkdir_p(File.join(root, "data"))
    panels.each { |name, rows| write_rows(File.join(root, "data", name), rows) }
    loaded.fetch(:tokenizer).save(File.join(root, "data/tokenizer.json"))
    sources = history_files.merge(Dir.glob("data/decision/downloads/semantics/*").select { |path| File.file?(path) }.to_h { |path| [File.expand_path(path), sha(path)] })
    protocol = { "version" => 3, "created_at" => Time.now.utc.iso8601, "config" => cfg.to_h,
      "parent" => PARENT, "parent_sha256" => loaded.fetch(:weights_fingerprint), "tokenizer" => loaded.fetch(:tokenizer).fingerprint,
      "parent_source" => "runs/decision/release-v01-corrective/selected", "control_protocol_sha256" => sha(File.join(CONTROL, "protocol.json")),
      "control_train_sha256" => old.fetch("files_sha256").fetch("train.jsonl"), "source_sha256" => sources,
      "files_sha256" => Dir.glob(File.join(root, "data/*")).to_h { |path| [File.basename(path), sha(path)] },
      "counts" => panels.transform_values { |rows| rows.group_by { |row| "#{row.fetch('source')}/#{row.fetch('language')}" }.transform_values(&:size) },
      "length_audit" => length_audit, "steps" => STEPS, "seed" => 1337, "arms" => ARMS,
      "sampling" => { "control" => "v0.2: 50% natural/unknown, 30% fact pairs, 20% routing",
        "candidate" => EasyAI::Decision::Data::RobustSampler::SIGNATURE },
      "fit" => { "maximum_steps" => 1000, "accuracy" => 0.99, "groups" => 0.95, "stable_checks" => 2 },
      "selection" => { "routing_tolerance" => 0.03, "minimum_known_accuracy" => 0.6, "minimum_unknown_accuracy" => 0.5,
        "minimum_binding_group_accuracy" => 0.4, "minimum_mixed_binding_group_accuracy" => 0.4, "minimum_wording_accuracy" => 0.5,
        "primary" => "Source/language macro factual accuracy, tie-break factual NLL" },
      "confirmation" => { "seed" => 2027, "minimum_parent_gain" => 0.03, "minimum_control_gain" => 0.01 },
      "publication" => { "minimum_parent_gain" => 0.03, "minimum_natural_gain" => 0.0, "maximum_routing_regression" => 0.03,
        "minimum_known_accuracy" => 0.6, "minimum_unknown_accuracy" => 0.5, "minimum_binding_group_accuracy" => 0.4,
        "minimum_mixed_binding_group_accuracy" => 0.4, "minimum_wording_accuracy" => 0.5, "confirmation_required" => true },
      "scope" => "Own v0.1 weights, unchanged shared bilingual model, CE control versus corrected supervision package; no teacher, RL, growth or business integration",
      "freshness" => "Natural panels exclude whole historical components; controlled panels have new expression and world families. Historical tests/routing are regression only." }
    write(File.join(root, "protocol.json"), protocol)
    puts JSON.pretty_generate(protocol.slice("counts", "length_audit", "selection"))
  end

  def fresh_natural(historical, collator)
    grouped = EasyAI::Decision::Data::SemanticCorpus.new(EasyAI::Decision::Data::SemanticAdapter.each("data/decision/downloads/semantics").to_a).split_rows
    seen = DecisionFactual.historical_components(grouped, historical)
    fresh = grouped.select { |_, group, split| split == "test" && !seen.include?(group) }
    buckets = fresh.group_by { |_, group, _| DecisionFactual.digest("v03-natural:#{group}").to_i(16) % 5 }
    counts = Hash.new(0)
    panels = %w[validation calibration test].each_with_index.to_h do |split, index|
      selected = index == 2 ? buckets.values_at(2, 3, 4).compact.flatten(1) : buckets.fetch(index, [])
      rows = selected.map { |row, group, _| DecisionFactual.convert(row, group) }
      supported = DecisionFactual.select_supported(rows, per_source: split == "test" ? 200 : 64, collator: collator, counts: counts)
      sources = supported.group_by { |row| row.fetch("source") }
      raise "Insufficient fresh natural material for #{split}" unless sources.size == 3 && sources.values.all? { |items| items.size >= 50 }
      [split, supported]
    end
    [panels, counts]
  end

  def fitting_rows(rows)
    controlled = rows.select { |row| row.fetch("source").start_with?("Factual-V03-") }
    groups = controlled.map { |row| row.fetch("group_id") }.uniq
    # Four complete worlds: two events with mixed/same facts and both languages.
    prefix = EasyAI::Decision::Data::RobustFacts::FAMILY_VERSION
    chosen = ["#{prefix}:train:transport:0", "#{prefix}:train:transport:1", "#{prefix}:train:work:2", "#{prefix}:train:work:3"]
    raise "Fitting groups absent" unless (chosen - groups).empty?
    extra = rows.reject { |row| row.fetch("source").start_with?("Factual-") }
      .group_by { |row| [row.fetch("source"), row.fetch("language")] }.values.flat_map { |items| DecisionFactual.stable(items, "v03-fit").first(4) }
    controlled.select { |row| chosen.include?(row.fetch("group_id")) } + extra
  end

  def audit_splits!(panels, historical, collator)
    train = panels.values_at("control-train.jsonl", "candidate-train.jsonl").flatten(1)
    evaluation = panels.values_at("validation.jsonl", "calibration.jsonl", "test.jsonl")
    sets = [train, *evaluation].map { |rows| rows.map { |row| row.fetch("group_id") }.to_set }
    sets.combination(2).each { |left, right| raise "Semantic group leakage" unless (left & right).empty? }
    materials = [train, *evaluation].map { |rows| rows.map { |row| DecisionFactual.material(row.fetch("state")) }.to_set }
    materials.combination(2).each { |left, right| raise "State material leakage" unless (left & right).empty? }
    evaluation.each do |rows|
      controlled = rows.select { |row| row.fetch("source").start_with?("Factual-V03-") }
      raise "Controlled historical material overlap" if controlled.any? { |row| historical.include?(DecisionFactual.material(row.fetch("state"))) }
    end
    panels.each do |name, rows|
      raise "Training rows outside bounds" if name.end_with?("-train.jsonl") && !(20_000..40_000).cover?(rows.size)
      raise "Duplicate row IDs: #{name}" unless rows.map { |row| row.fetch("id") }.uniq.size == rows.size
      rows.each do |row|
        EasyAI::Decision::Data::Example.new(row)
        next if name == "regression.jsonl"
        collator.state_tokens(row.fetch("state"))
        row.fetch("options").each { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }
      end
    end
  end
end
