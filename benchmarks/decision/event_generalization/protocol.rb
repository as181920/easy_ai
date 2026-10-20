module DecisionEventGeneralization
  module_function

  def seal(path, value)
    raise "Preserve completed file: #{path}" if File.exist?(path)
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    value
  end

  def sealed(path)
    raise "Sealed artifact changed: #{path}" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    JSON.parse(File.read(path))
  end

  def parent(seed)
    result = JSON.parse(File.read(File.join(ROOT_V07, "arms", "combined-#{seed}", "result-6000.json")))
    result.fetch("checkpoint")
  end

  def raw(root, name)
    DecisionRelational.rows(File.join(root, "data", "#{name}.jsonl"))
  end

  def ledger(root)
    Dir[File.join(root, "{preflight,arms}/**/training.jsonl")].sort.to_h { |file| [file.delete_prefix(root + "/"), File.foreach(file).count] }
  end

  def budget!(root, additional = 0)
    raise "v08 actual-update ceiling exceeded" if ledger(root).values.sum + additional > CEILING
  end

  def prepare(root)
    raise "Preserve existing v08 protocol" if File.exist?(File.join(root, "protocol.json"))
    DecisionBindingGeneralization.verify(ROOT_V07)
    FileUtils.mkdir_p(File.join(root, "data"))
    replay_root = File.expand_path("runs/decision/event-generalization-v08")
    replay_families = JSON.parse(File.read(File.join(replay_root, "replayed-augmented-families.json")))
    replay_materials = JSON.parse(File.read(File.join(replay_root, "replayed-augmented-materials.json")))
    %w[reference coverage].each do |variant|
      proof = JSON.parse(File.read(File.join(replay_root, "replay-proof-#{variant}.json")))
      checkpoint = EasyAI::Decision::Checkpoint.resolve(File.join(replay_root, "arms", "#{variant}-1337", "choice"))
      raise "Prefix replay proof changed" unless proof.fetch("step") == 1500 && proof.fetch("metadata_sha256") == sha(File.join(checkpoint, "metadata.json"))
    end
    originals = Dir["runs/decision/**/data/*.jsonl"].reject { |file| File.expand_path(file).start_with?(root + "/") }.sort.to_h { |file| [file, sha(file)] }
    materials, families = Set.new, Set.new
    old_names = Catalog::LANGUAGES.to_h { |language| [language, Set.new] }
    originals.group_by { |_, digest| digest }.each_value do |copies|
      File.foreach(copies.first.first) do |line|
        row = JSON.parse(line)
        materials << Catalog.material(row.fetch("state")) if row["state"]
        next unless row.dig("world", "facts").is_a?(Array)
        family = Catalog.family(row)
        families << family if family
        facts = row.fetch("world").fetch("facts")
        facts.each { |fact| old_names[row["language"]]&.add(fact.fetch("actor")) } if facts.all? { |fact| fact.is_a?(Hash) && fact.key?("actor") }
      end
    end
    families.merge(replay_families)
    materials.merge(replay_materials)
    Catalog::LANGUAGES.each do |language|
      names = %w[development lockbox].flat_map { |split| Catalog.novel_names(language, split) }
      raise "Novel identities previously used" unless (names.to_set & old_names.fetch(language)).empty?
    end
    development = Corpus.panels(:development, worlds: 32, excluded_families: families, historical_materials: materials)
    development.values.flatten.each { |row| families << Catalog.family(row) }
    lockbox = Corpus.panels(:lockbox, worlds: 64, excluded_families: families, historical_materials: materials)
    all = development.transform_keys { |name| "development-#{name}" }.merge(lockbox.transform_keys { |name| "lockbox-#{name}" })
    all.each do |name, rows|
      audit_rows!(rows)
      DecisionRobust.write_rows(File.join(root, "data", "#{name}.jsonl"), rows)
    end
    %w[train probe].each { |name| FileUtils.cp(File.join(ROOT_V07, "data", "#{name}.jsonl"), File.join(root, "data", "#{name}.jsonl")) }
    %w[canonical seen_combinations novel_actor].each do |name|
      FileUtils.cp(File.join(ROOT_V07, "data", "#{name}.jsonl"), File.join(root, "data", "regression-#{name}.jsonl"))
    end
    forbidden = reserved_materials(all.values.flatten)
    forbidden.merge(JSON.parse(File.read(File.join(replay_root, "forbidden.json"))))
    Dir[File.join(DecisionBindingGeneralization::ROOT_V06, "data/*.jsonl")].reject { |file| File.basename(file).start_with?("train", "probe") }.each do |file|
      File.foreach(file) { |line| forbidden << Catalog.material(JSON.parse(line).fetch("state")) }
    end
    write(File.join(root, "forbidden.json"), forbidden.to_a.sort)
    evidence_directory = File.join(root, "isolation-correction")
    FileUtils.mkdir_p(evidence_directory)
    %w[family-isolation-discovery.json family-replay-audit.json replay-proof-reference.json replay-proof-coverage.json replayed-augmented-families.json replayed-augmented-materials.json replay-family-audit.rb].each do |name|
      FileUtils.cp(File.join(replay_root, name), File.join(evidence_directory, name))
    end
    FileUtils.mkdir_p(File.join(root, "preflight"))
    FileUtils.cp_r(File.join(replay_root, "preflight"), File.join(root, "preflight", "inherited"))
    examples = raw(root, "probe").map { |row| EasyAI::Decision::Data::Example.new(row) }
    augmented = Augmentation.new(rows: raw(root, "probe"), forbidden: forbidden).call(examples, rng: Random.new(818_081))
    DecisionRobust.write_rows(File.join(root, "data", "probe-augmented.jsonl"), augmented)
    sources = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort.to_h { |file| [file, sha(file)] }
    sources.each_key do |file|
      destination = File.join(root, "source-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
    end
    parents = [1337, 2027].to_h { |seed| [seed, { "path" => parent(seed), "weights_sha256" => sha(File.join(parent(seed), "weights.pt")) }] }
    loaded = EasyAI::Decision::Checkpoint.load(parent(1337))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config)
    maxima = { "state" => 0, "option" => 0 }
    Dir[File.join(root, "data/*.jsonl")].each do |file|
      File.foreach(file) do |line|
        row = JSON.parse(line)
        maxima["state"] = [maxima.fetch("state"), collator.state_tokens(row.fetch("state")).size].max
        row.fetch("options").each { |option| maxima["option"] = [maxima.fetch("option"), collator.option_tokens(row.fetch("question"), option.fetch("text")).size].max }
      end
    end
    raise "v08 data truncation" unless collator.truncated.zero?
    FileUtils.cp("docs/decision/v08-plan.md", File.join(root, "plan-frozen.md"))
    protocol = { "version" => 8, "created_at" => Time.now.utc.iso8601, "parents" => parents, "source_sha256" => sources,
      "files_sha256" => Dir[File.join(root, "data/*.jsonl")].sort.to_h { |file| [File.basename(file), sha(file)] },
      "forbidden_sha256" => sha(File.join(root, "forbidden.json")), "plan_sha256" => sha(File.join(root, "plan-frozen.md")),
      "catalog" => Catalog::EVENTS, "event_pools" => Catalog::POOLS, "form_pools" => Catalog::FORMS,
      "isolation_correction" => { "retired_root" => replay_root, "retired_protocol_sha256" => sha(File.join(replay_root, "protocol.json")),
        "evidence_sha256" => Dir[File.join(evidence_directory, "*")].to_h { |file| [File.basename(file), sha(file)] },
        "scope" => "Original panel retired after cross-template augmented-family overlap. Replacement uses unchanged catalog, split sizes, seeds and selection rules, with exclusion of actual parent and prefix augmented families/materials. No replacement-panel predictions precede freezing. Exact prefix rendering replay is checked; valid prefixes are retained and all physical/replayed updates counted once." },
      "steps" => 6000, "primary_variants" => VARIANTS, "confirmation_seed" => 2027, "ceiling" => CEILING,
      "data_audit" => { "historical_files" => originals, "historical_materials" => materials.size, "novel_name_overlap" => 0,
        "development_rows" => development.values.sum(&:size), "lockbox_rows" => lockbox.values.sum(&:size), "token_maxima" => maxima,
        "truncations" => collator.truncated, "development_worlds_per_language_view" => 32, "lockbox_worlds_per_language_view" => 64,
        "development_mixed_pairs_per_axis_language_view" => 128, "lockbox_mixed_pairs_per_axis_language_view" => 256,
        "scope" => "Exact and normalized historical material exclusion; unordered actor/event world-family exclusion across frozen panels. Novel-person identity check covers structured historical fact metadata, not arbitrary parent pretraining." },
      "scope" => "Binary synthetic event/expression direction experiment with fixed own v07 parents and fresh development/lockbox. No factual release, no unknown/natural QA or scaling claim." }
    seal(File.join(root, "protocol.json"), protocol)
    puts "Frozen v08: 10240 development + 20480 unopened lockbox rows; maxima #{maxima}"
  end

  def verify(root)
    protocol = sealed(File.join(root, "protocol.json"))
    protocol.fetch("parents").each_value { |value| raise "v08 parent changed" unless sha(File.join(value.fetch("path"), "weights.pt")) == value.fetch("weights_sha256") }
    protocol.fetch("files_sha256").each { |file, digest| raise "v08 data changed: #{file}" unless sha(File.join(root, "data", file)) == digest }
    raise "v08 exclusion set changed" unless sha(File.join(root, "forbidden.json")) == protocol.fetch("forbidden_sha256")
    protocol.fetch("isolation_correction").fetch("evidence_sha256").each { |file, digest| raise "Isolation evidence changed" unless sha(File.join(root, "isolation-correction", file)) == digest }
    # The CLI and read-only reporting may evolve; training/data/policy sources remain frozen.
    protocol.fetch("source_sha256").each do |file, digest|
      next if ["benchmarks/decision/event_generalization.rb", "benchmarks/decision/event_generalization/report.rb", "benchmarks/decision/event_generalization/audit.rb"].include?(file)
      raise "v08 frozen implementation changed: #{file}" unless sha(file) == digest
    end
    protocol
  end

  def trainer(root, variant, seed, output, resume: false)
    forbidden = JSON.parse(File.read(File.join(root, "forbidden.json")))
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/train.jsonl"))
    if resume
      loaded = EasyAI::Decision::Checkpoint.load(output)
      migration = nil
      if loaded.fetch(:metadata).dig("training", "event_generalization", "forbidden_sha256") != Digest::SHA256.hexdigest(JSON.generate(forbidden.sort))
        migration = sealed(File.join(File.dirname(output), "prefix-migration.json"))
        raise "Prefix weights changed before migration" unless migration.fetch("weights_sha256") == loaded.fetch(:weights_fingerprint)
      end
      return Trainer.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output, variant: variant,
        forbidden_states: forbidden, restored: loaded.fetch(:metadata), migration: migration, device: :cuda)
    end
    loaded = EasyAI::Decision::Checkpoint.load(parent(seed))
    config = DecisionScaling.model_config(loaded.fetch(:model), "base", microbatch: 16, steps: 6000, seed: seed)
    model, = EasyAI::Decision::WeightTransfer.apply(parent: loaded, config: config, tokenizer: loaded.fetch(:tokenizer), seed: seed)
    Trainer.new(model: model, tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output, variant: variant, forbidden_states: forbidden, device: :cuda)
  end

  def reserved_materials(rows)
    rows.flat_map { |row| (0..6).map { |form| Catalog.material(Catalog.render(row, form: form).fetch("state")) } }.to_set
  end

  def import_prefixes(root)
    protocol = verify(root)
    source = protocol.fetch("isolation_correction").fetch("retired_root")
    forbidden = JSON.parse(File.read(File.join(root, "forbidden.json")))
    VARIANTS.each do |variant|
      destination = File.join(root, "arms", "#{variant}-1337")
      next if File.exist?(File.join(destination, "prefix-migration.json")) && sealed(File.join(destination, "prefix-migration.json"))
      old = File.join(source, "arms", "#{variant}-1337", "choice")
      checkpoint = EasyAI::Decision::Checkpoint.resolve(old)
      metadata = JSON.parse(File.read(File.join(checkpoint, "metadata.json"))).fetch("training")
      raise "Unsafe imported prefix step" unless metadata.fetch("step") == 1500
      raise "Prefix overlaps replacement reserved worlds" unless (metadata.fetch("rendering_coverage").fetch("surface_hashes").keys.to_set & forbidden.to_set).empty?
      FileUtils.mkdir_p(destination)
      FileUtils.cp_r(old, File.join(destination, "choice"))
      seal(File.join(destination, "prefix-migration.json"), { "previous_signature" => metadata.fetch("event_generalization"),
        "new_forbidden_sha256" => Digest::SHA256.hexdigest(JSON.generate(forbidden.sort)), "weights_sha256" => sha(File.join(checkpoint, "weights.pt")),
        "checkpoint_step" => 1500, "trace_sha256" => sha(File.join(old, "training.jsonl")), "rendered_state_overlap" => 0,
        "scope" => "Strengthened exclusions after exact rendering-history audit and replacement-panel freezing; model/optimizer/sampler state unchanged. Retained trace includes discarded-tail executions; resume from committed 1500 and count replay." })
    end
  end
end
