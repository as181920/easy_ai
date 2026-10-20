module DecisionBindingGeneralization
  module_function

  def held_state_hashes(row)
    language = row.fetch("language")
    clauses = row.fetch("world").fetch("facts").map do |fact|
      phrase = EasyAI::Decision::Data::FactualContrasts::EVENTS.fetch(fact.fetch("event")).fetch((language == "zh-CN" ? 0 : 2) + (fact.fetch("truth") ? 0 : 1))
      language == "zh-CN" ? fact.fetch("actor") + phrase : fact.fetch("actor") + " " + phrase
    end
    [clauses, clauses.reverse].map do |ordered|
      rendered = ordered.join(language == "zh-CN" ? "。" : ". ") + (language == "zh-CN" ? "。" : ".")
      Digest::SHA256.hexdigest(rendered)
    end.uniq
  end

  def prepare(root)
    raise "Preserve existing v07 experiment" if File.exist?(File.join(root, "protocol.json"))
    DecisionScaling.verify(ROOT_V06)
    FileUtils.mkdir_p(File.join(root, "data"))
    sources = { "train" => "train-128-binary", "probe" => "probe-binary", "canonical" => "validation",
      "seen_combinations" => "validation-seen_combinations", "novel_actor" => "validation-novel_actor", "novel_event" => "validation-novel_event" }
    panels = sources.to_h do |name, original|
      values = DecisionScaling.binary_rows(DecisionRelational.rows(File.join(ROOT_V06, "data", "#{original}.jsonl")))
      EasyAI::Decision::Data::QualityAudit.integrity!(values)
      DecisionRobust.write_rows(File.join(root, "data", "#{name}.jsonl"), values)
      [name, values]
    end
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(panels.fetch("train"), *panels.except("train", "probe").values)
    forbidden = Dir[File.join(ROOT_V06, "data/*.jsonl")].reject { |path| File.basename(path).start_with?("train", "probe") }.flat_map do |path|
      File.foreach(path).flat_map { |line| held_state_hashes(JSON.parse(line)) }
    end.uniq.sort
    write(File.join(root, "forbidden-state-hashes.json"), forbidden)
    files = Dir[File.join(root, "data/*.jsonl")].to_h { |path| [File.basename(path), sha(path)] }
    source = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort.to_h { |file| [file, sha(file)] }
    source.each_key do |file|
      destination = File.join(root, "source-frozen", file)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(file, destination)
    end
    value = { "version" => 7, "created_at" => Time.now.utc.iso8601, "parent" => DecisionScaling::PARENT,
      "parent_weights_sha256" => sha(File.join(DecisionScaling::PARENT, "weights.pt")),
      "previous_protocol_sha256" => sha(File.join(ROOT_V06, "protocol.json")), "files_sha256" => files,
      "forbidden_sha256" => sha(File.join(root, "forbidden-state-hashes.json")), "source_sha256" => source,
      "primary_variants" => VARIANTS, "steps" => 6000, "worlds_per_language" => 128,
      "data_rows" => panels.fetch("train").size, "seed" => 1337, "evaluation_every" => 500,
      "mastery" => { "known_balanced" => 0.95, "dev_balanced" => 0.55, "mixed_groups" => 0.85, "stable_checks" => 2 },
      "direction" => { "worst_actor_gain" => 0.05, "maximum_balanced_drop" => 0.03 },
      "update_ceiling" => UPDATE_CEILING, "sampler" => Trainer::SIGNATURE,
      "extension" => { "variant" => "ce", "maximum_steps" => 12000, "training_accuracy_gain" => 0.01, "training_group_gain" => 0.03 },
      "conditional" => "Four-arm rename x pair-loss factorial. Confirm an improving selected recipe and its CE reference at seed 2027. If none improves, diagnose a frozen pretrained encoder; record a separate protocol before download/evaluation.",
      "scope" => "Own-parent 6.63M model, 128 worlds, effective batch 32/micro16, LR .0003, no dropout/warmup. Historical v06 base32/1500 is equal-mean-exposure reference. Observed development only; no release, unknown-stage training or acceptance selection. Rename pool excludes novel actors and rejects all reserved material hashes. Matched visits are not matched FLOPs." }
    FileUtils.cp("docs/decision/v07-plan.md", File.join(root, "plan-frozen.md"))
    value["plan_sha256"] = sha(File.join(root, "plan-frozen.md"))
    write(File.join(root, "protocol.json"), value)
    write(File.join(root, "protocol.sha256.json"), { "sha256" => sha(File.join(root, "protocol.json")) })
    puts "Frozen v07: #{value.fetch('data_rows')} rows, four independent 6000-step arms"
  end

  def verify(root)
    path = File.join(root, "protocol.json")
    raise "v07 protocol changed" unless sha(path) == JSON.parse(File.read(File.join(root, "protocol.sha256.json"))).fetch("sha256")
    value = JSON.parse(File.read(path))
    raise "v07 parent changed" unless sha(File.join(value.fetch("parent"), "weights.pt")) == value.fetch("parent_weights_sha256")
    value.fetch("files_sha256").each { |name, digest| raise "v07 data changed" unless sha(File.join(root, "data", name)) == digest }
    raise "v07 exclusions changed" unless sha(File.join(root, "forbidden-state-hashes.json")) == value.fetch("forbidden_sha256")
    sources = value.fetch("source_sha256")
    previous_amendment = nil
    %w[namespace-correction/amendment exposure-confirmation/amendment selection-integrity/amendment reporting-integrity/amendment].each do |name|
      amendment_path = File.join(root, name + ".json")
      next unless File.exist?(amendment_path)
      raise "Runtime amendment changed" unless sha(amendment_path) == JSON.parse(File.read(amendment_path + ".sha256.json")).fetch("sha256")
      amendment = JSON.parse(File.read(amendment_path))
      raise "Runtime amendment protocol mismatch" unless amendment.fetch("protocol_sha256") == sha(path)
      if previous_amendment && amendment.fetch("previous_amendment_sha256") != previous_amendment
        raise "Runtime amendment chain mismatch"
      end
      sources = amendment.fetch("runtime_source_sha256")
      previous_amendment = sha(amendment_path)
    end
    sources.each do |file, digest|
      next if file == "benchmarks/decision/binding_generalization.rb"
      raise "Frozen v07 learning implementation changed: #{file}" unless sha(file) == digest
    end
    value
  end

  def ledger(root)
    Dir[File.join(root, "{preflight,arms}/**/training.jsonl")].sort.to_h do |path|
      [path.delete_prefix(root + "/"), File.foreach(path).count]
    end
  end

  def budget!(root, additional)
    raise "v07 actual-update ceiling exceeded" if ledger(root).values.sum + additional > UPDATE_CEILING
  end

  def raw(root, name)
    DecisionRelational.rows(File.join(root, "data", "#{name}.jsonl"))
  end

  def trainer(root, variant, seed, output, resume: false)
    config = DecisionScaling.model_config(EasyAI::Decision::Checkpoint.load(DecisionScaling::PARENT).fetch(:model), "base", microbatch: 16, steps: 12000, seed: seed)
    dataset = EasyAI::Decision::Data::Dataset.new(File.join(root, "data/train.jsonl"))
    forbidden = JSON.parse(File.read(File.join(root, "forbidden-state-hashes.json")))
    if resume
      loaded = EasyAI::Decision::Checkpoint.load(output)
      raise "v07 resume config changed" unless loaded.fetch(:model).config.to_h == config.to_h
      raise "v07 terminal checkpoint cannot be silently resumed" if loaded.fetch(:metadata).fetch("training")["stop_reason"]
      Trainer.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output,
        restored: loaded.fetch(:metadata), variant: variant, forbidden_states: forbidden)
    else
      model, tokenizer, = DecisionScaling.initialize_model("base", microbatch: 16, steps: 12000, seed: seed)
      Trainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: output, variant: variant, forbidden_states: forbidden)
    end
  end
end
