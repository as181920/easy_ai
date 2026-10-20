module DecisionExpressionRetention
  module_function

  def ledger(root)
    Dir[File.join(root, "{preflight,arms}/**/training.jsonl")].sort.to_h { |file| [file.delete_prefix(root + "/"), File.foreach(file).count] }
  end

  def budget!(root, additional = 0)
    raise "v09 actual-update ceiling exceeded" if ledger(root).values.sum + additional > CEILING
  end

  def audit_rows!(rows)
    raise "Duplicate v09 IDs" unless rows.map { |row| row.fetch("id") }.uniq.size == rows.size
    seen = {}
    rows.each do |row|
      EasyAI::Decision::Data::Example.new(row)
      Corpus.gold!(row)
      world = row.fetch("world")
      raise "Invalid explicit truth" unless world.fetch("facts").all? { |fact| [true, false].include?(fact.fetch("truth")) } && [true, false].include?(world.fetch("assertion"))
      raise "Invalid mixed metadata" unless world.fetch("mixed") == (world.fetch("facts").map { |fact| fact.fetch("truth") }.uniq.size == 2)
      rendered = Catalog.render(row, form: world.fetch("form", 0))
      raise "Renderer mismatch" unless rendered.values_at("state", "question") == row.values_at("state", "question")
      key = Digest::SHA256.hexdigest(JSON.generate(row.values_at("state", "question", "options")))
      raise "Conflicting same inputs" if seen.key?(key) && seen.fetch(key) != row.fetch("target")
      seen[key] = row.fetch("target")
    end
    %w[binding fact_flip question_flip].each do |axis|
      raise "Incomplete counterfactual" unless rows.group_by { |row| row.fetch("contrast_groups").fetch(axis) }.values.all? { |group| group.size == 2 }
    end
    true
  end

  def prepare(root)
    raise "Preserve existing preparation" if File.exist?(File.join(root, "protocol.json"))
    DecisionEventGeneralization.verify(ROOT_V08)
    historical = Dir["runs/decision/**/data/*.jsonl"].reject { |file| File.expand_path(file).start_with?(root + "/") }.sort.to_h { |file| [file, sha(file)] }
    materials, families = Set.new, Set.new
    names = Catalog::LANGUAGES.to_h { |language| [language, Set.new] }
    historical.group_by { |_, digest| digest }.each_value do |copies|
      File.foreach(copies.first.first) do |line|
        row = JSON.parse(line)
        materials << Catalog.material(row.fetch("state")) if row["state"]
        next unless row.dig("world", "facts").is_a?(Array)
        family = Catalog.family(row)
        families << family if family
        row.dig("world", "facts").each { |fact| names[row["language"]]&.add(fact.fetch("actor")) } if family
      end
    end
    replay_path = File.join(ROOT_V08, "isolation-correction/replayed-augmented-families.json")
    replay_materials = File.join(ROOT_V08, "isolation-correction/replayed-augmented-materials.json")
    families.merge(JSON.parse(File.read(replay_path)))
    materials.merge(JSON.parse(File.read(replay_materials)))
    Catalog::LANGUAGES.each do |language|
      novel = %w[development lockbox].flat_map { |split| Catalog.novel_names(language, split) }
      raise "Novel identity reused" unless (novel.to_set & names.fetch(language)).empty?
    end
    dev = Corpus.panels(:development, worlds: 32, excluded_families: families, historical_materials: materials)
    dev.values.flatten.each { |row| families << Catalog.family(row) }
    final = Corpus.panels(:lockbox, worlds: 64, excluded_families: families, historical_materials: materials)
    panels = dev.transform_keys { |name| "development-#{name}" }.merge(final.transform_keys { |name| "lockbox-#{name}" })
    FileUtils.mkdir_p(File.join(root, "data"))
    panels.each do |name, rows|
      audit_rows!(rows)
      DecisionRobust.write_rows(File.join(root, "data", "#{name}.jsonl"), rows)
    end
    %w[train probe].each { |name| FileUtils.cp(File.join(DecisionEventGeneralization::ROOT_V07, "data", "#{name}.jsonl"), File.join(root, "data", "#{name}.jsonl")) }
    mapping = EasyAI::Decision::Data::FactualContrasts::TRAIN_EVENTS.zip(Catalog::POOLS.fetch("train")).to_h
    augmented_probe = raw(root, "probe").map do |row|
      facts = row.dig("world", "facts").map { |fact| fact.merge("event" => mapping.fetch(fact.fetch("event"))) }
      Catalog.render(row, facts: facts, event: mapping.fetch(row.dig("world", "query_event")), form: 7)
    end
    DecisionRobust.write_rows(File.join(root, "data/probe-augmented.jsonl"), augmented_probe)
    reserved = panels.values.flatten
    Dir[File.join(DecisionEventGeneralization::ROOT_V07, "data/*.jsonl")].reject { |file| File.basename(file).start_with?("train", "probe") }.each do |file|
      reserved.concat(DecisionRelational.rows(file))
    end
    Catalog::VIEWS.each do |view|
      rows = raw(ROOT_V08, "lockbox-#{view}")
      if %w[expressions composed].include?(view)
        crossed = DecisionRelational.rows(File.join(ROOT_V08, "crossed-template-diagnostic", "baseline-1337-#{view}.jsonl")).map { |row| row.except("prediction", "logits") }
        rows += crossed.map do |row|
          row.merge("id" => "#{row.fetch('id')}:crossed", "group_id" => "#{row.fetch('group_id')}:crossed", "contrast_groups" => row.fetch("contrast_groups").transform_values { |value| "#{value}:crossed" })
        end
      end
      DecisionRobust.write_rows(File.join(root, "data", "regression-#{view}.jsonl"), rows)
      reserved.concat(rows)
    end
    forbidden = reserved.map { |row| Catalog.family(row) }.to_set
    states = reserved.flat_map { |row| (0..12).map { |form| Catalog.material(Catalog.render(row, form: form).fetch("state")) } }.to_set
    states.merge(JSON.parse(File.read(File.join(DecisionEventGeneralization::ROOT_V07, "forbidden-state-hashes.json"))))
    write(File.join(root, "forbidden-families.json"), forbidden.to_a.sort)
    write(File.join(root, "forbidden-states.json"), states.to_a.sort)
    sources = (Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"] + ["test/easy_ai/decision/expression_retention_test.rb"]).sort.to_h { |file| [file, sha(file)] }
    sources.each_key do |file|
      target = File.join(root, "source-frozen", file)
      FileUtils.mkdir_p(File.dirname(target))
      FileUtils.cp(file, target)
    end
    FileUtils.cp("docs/decision/v09-plan.md", File.join(root, "plan-frozen.md"))
    loaded = EasyAI::Decision::Checkpoint.load(parent(1337))
    other = EasyAI::Decision::Checkpoint.load(parent(2027))
    raise "Tokenizer mismatch" unless loaded.fetch(:tokenizer).fingerprint == other.fetch(:tokenizer).fingerprint
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config)
    maxima = { "state" => 0, "option" => 0 }
    inputs = {}
    counts = Dir[File.join(root, "data/*.jsonl")].to_h do |file|
      count = 0
      File.foreach(file) do |line|
        row = JSON.parse(line)
        Corpus.gold!(row)
        ids = [collator.state_tokens(row.fetch("state")), row.fetch("options").map { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }]
        maxima["state"] = [maxima.fetch("state"), ids.first.size].max
        maxima["option"] = [maxima.fetch("option"), ids.last.map(&:size).max].max
        digest = Digest::SHA256.hexdigest(JSON.generate(ids))
        raise "Tokenized gold conflict" if inputs.key?(digest) && inputs.fetch(digest) != row.fetch("target")
        inputs[digest] = row.fetch("target")
        count += 1
      end
      [File.basename(file), count]
    end
    raise "Truncated preparation" unless collator.truncated.zero?
    seal(File.join(root, "protocol.json"), { "version" => 9, "created_at" => Time.now.utc.iso8601, "variants" => VARIANTS, "steps" => STEPS, "ceiling" => CEILING,
      "parents" => [1337, 2027].to_h { |seed| [seed, { "path" => parent(seed), "weights_sha256" => sha(File.join(parent(seed), "weights.pt")) }] },
      "files_sha256" => Dir[File.join(root, "data/*.jsonl")].sort.to_h { |file| [File.basename(file), sha(file)] },
      "family_exclusions_sha256" => sha(File.join(root, "forbidden-families.json")), "state_exclusions_sha256" => sha(File.join(root, "forbidden-states.json")),
      "source_sha256" => sources, "plan_sha256" => sha(File.join(root, "plan-frozen.md")), "tokenizer_fingerprint" => loaded.fetch(:tokenizer).fingerprint,
      "catalog" => Catalog::EVENTS, "event_pools" => Catalog::POOLS, "form_pools" => Catalog::FORMS,
      "data_audit" => { "rows_by_file" => counts, "unique_tokenized_inputs" => inputs.size, "token_maxima" => maxima, "truncations" => 0, "tokenized_gold_collisions" => 0,
        "historical_files" => historical, "historical_materials" => materials.size, "allocation_excluded_families" => families.size, "reserved_families" => forbidden.size,
        "parent_family_evidence_sha256" => sha(replay_path), "parent_material_evidence_sha256" => sha(replay_materials), "novel_identity_overlap" => 0,
        "scope" => "World-family isolation from actual v07 parent/prefix augmentation and all historical raw panels. Later unrelated v08 branch augmentation does not initialize v09 and is not fully replayed. New wording is crossed within every held family on both axes; repeated forms are not independent worlds." },
      "scope" => "Own6.63M binary synthetic event-versus-expression coverage and canonical replay test. No natural QA, unknown, scaling or factual-release claim." })
    puts "Frozen v09: #{counts}, maxima=#{maxima}; final predictions remain unopened"
  end

  def verify(root)
    value = sealed(File.join(root, "protocol.json"))
    value.fetch("parents").each_value { |parent| raise "Parent changed" unless sha(File.join(parent.fetch("path"), "weights.pt")) == parent.fetch("weights_sha256") }
    value.fetch("files_sha256").each { |file, digest| raise "Data changed #{file}" unless sha(File.join(root, "data", file)) == digest }
    raise "Family exclusions changed" unless sha(File.join(root, "forbidden-families.json")) == value.fetch("family_exclusions_sha256")
    raise "State exclusions changed" unless sha(File.join(root, "forbidden-states.json")) == value.fetch("state_exclusions_sha256")
    raise "Plan archive changed" unless sha(File.join(root, "plan-frozen.md")) == value.fetch("plan_sha256")
    value.fetch("source_sha256").each do |file, digest|
      next if file == "benchmarks/decision/expression_retention/presentation.rb"
      raise "Frozen implementation changed #{file}" unless sha(file) == digest
    end
    value
  end

  def trainer(root, variant, seed, output, resume: false)
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/train.jsonl"))
    options = { dataset: dataset, output: output, variant: variant, device: :cuda,
      forbidden_families: JSON.parse(File.read(File.join(root, "forbidden-families.json"))), forbidden_states: JSON.parse(File.read(File.join(root, "forbidden-states.json"))) }
    if resume
      loaded = EasyAI::Decision::Checkpoint.load(output)
      return Trainer.new(**options, model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), restored: loaded.fetch(:metadata))
    end
    loaded = EasyAI::Decision::Checkpoint.load(parent(seed))
    config = DecisionScaling.model_config(loaded.fetch(:model), "base", microbatch: 16, steps: STEPS, seed: seed)
    model, = EasyAI::Decision::WeightTransfer.apply(parent: loaded, config: config, tokenizer: loaded.fetch(:tokenizer), seed: seed)
    Trainer.new(**options, model: model, tokenizer: loaded.fetch(:tokenizer))
  end

  def replay(root, source)
    raise "Replay destination exists" if File.exist?(root)
    verify(source)
    FileUtils.mkdir_p(root)
    %w[data forbidden-families.json forbidden-states.json protocol.json protocol.json.sha256.json plan-frozen.md source-frozen].each { |name| FileUtils.cp_r(File.join(source, name), File.join(root, name)) }
    seal(File.join(root, "replay-marker.json"), { "source" => source, "scope" => "Observed panel replay, not a newly reserved independent evaluation." })
  end
end
