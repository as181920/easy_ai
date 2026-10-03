module DecisionJudgment
  module_function

  def canonical(row, binary: false, single: false)
    result = Marshal.load(Marshal.dump(row))
    result["options"] = EasyAI::Decision::Data::JudgmentCorpus.options(row.fetch("language"), binary: binary)
    result["id"] += single ? ":single" : binary ? ":binary" : ":canonical"
    result["contrast_groups"] = row.fetch("contrast_groups", {}).transform_values { |value| "#{value}:#{binary}:#{single}" }
    world = row.fetch("world")
    language = row.fetch("language")
    phrase = EasyAI::Decision::Data::FactualContrasts::EVENTS.fetch(world.fetch("event")).fetch((language == "zh-CN" ? 0 : 2) + (world.fetch("assertion") ? 0 : 1))
    claim = language == "zh-CN" ? "#{world.fetch('queried_actor')}#{phrase}" : "#{world.fetch('queried_actor')} #{phrase}"
    result["question"] = language == "zh-CN" ? "根据记录判断陈述：#{claim}。" : "Judge this claim against the record: #{claim}."
    if single
      actor = world.fetch("queried_actor")
      truth = world.fetch("facts").fetch(actor)
      phrase = EasyAI::Decision::Data::FactualContrasts::EVENTS.fetch(world.fetch("event")).fetch((row.fetch("language") == "zh-CN" ? 0 : 2) + (truth ? 0 : 1))
      result["state"] = row.fetch("language") == "zh-CN" ? "#{actor}#{phrase}。" : "#{actor} #{phrase}."
      result["source"] = "Factual-V04-Control-Single"
    end
    result
  end

  def control_rows
    rows = raw("runs/decision/factual-v03/pilot/data/candidate-train.jsonl").select { |row| row.dig("world", "version") == 3 && row.dig("world", "variant") == "base" }
    groups = rows.group_by { |row| row.fetch("group_id") }.sort_by { |group, _| sha_text("v04-control:#{group}") }.first(128).flat_map(&:last)
    groups.flat_map do |row|
      known = row.fetch("target") != "unknown"
      [canonical(row), *(known ? [canonical(row, binary: true), canonical(row, binary: true, single: true)] : [])]
    end
  end

  def sha_text(text)
    Digest::SHA256.hexdigest(text)
  end

  def prepare(root)
    raise "Output exists; retain prior evidence" if File.exist?(root)
    historical, _, history_files = DecisionRelease.history
    review_path = "runs/decision/judgment-v04/review/audit.json"
    audit = JSON.parse(File.read(review_path))
    # Disagreements remain unresolved, not automatically accepted relabels.
    natural = audit.fetch("decisions").select { |row| row.fetch("agreement") && !row.fetch("excluded") && %w[BoolQ OCNLI].include?(row.fetch("source")) && %w[yes no].include?(row.fetch("target")) }
      .map { |row| row.slice("id", "group_id", "language", "state", "question", "target", "source").merge("options" => EasyAI::Decision::Data::JudgmentCorpus.options(row.fetch("language"), binary: true)) }
    replay = raw("runs/decision/factual-v03/pilot/data/candidate-train.jsonl").select { |row| row.fetch("source") == "MASSIVE-Scenario" }
    controlled = EasyAI::Decision::Data::JudgmentCorpus.splits
    # Evaluation claims use both polarities; all truth assignments are kept.
    panels = controlled.reject { |name, _| name == "train" }.transform_keys { |name| "#{name}.jsonl" }
    panels["candidate-train.jsonl"] = controlled.fetch("train") + natural + replay
    panels["control-train.jsonl"] = control_rows + natural + replay
    # Each probe belongs to that arm's actual training families.
    ARMS.each do |arm|
      core = panels.fetch("#{arm}-train.jsonl").select { |row| row.fetch("source").start_with?("Factual-") && row.fetch("options").size == 2 && !row.fetch("source").end_with?("Single") }
      families = core.map { |row| row.fetch("group_id") }.uniq.first(4)
      panels["#{arm}-probe.jsonl"] = core.select { |row| families.include?(row.fetch("group_id")) }
    end
    # A complete training-only sample; pure fitting has no natural/routing mixture.
    fit_groups = controlled.fetch("train").map { |row| row.fetch("group_id") }.uniq.first(2)
    panels["fit.jsonl"] = controlled.fetch("train").select { |row| fit_groups.include?(row.fetch("group_id")) && row.fetch("options").size == 2 && !row.fetch("source").end_with?("Single") }
    fresh_review_path = "runs/decision/judgment-v04/review-natural/audit.json"
    fresh_review = JSON.parse(File.read(fresh_review_path))
    panels["natural-test.jsonl"] = fresh_review.fetch("decisions").select { |row| row.fetch("agreement") && !row.fetch("excluded") }
      .flat_map do |row|
        profiles = row.fetch("target") == "unknown" ? ["joint"] : %w[joint binary]
        profiles.map do |profile|
          row.slice("id", "group_id", "language", "state", "question", "target", "source").merge("id" => "#{row.fetch('id')}:#{profile}",
            "options" => EasyAI::Decision::Data::JudgmentCorpus.options(row.fetch("language"), binary: profile == "binary"), "profile" => profile)
        end
      end
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    cfg = loaded.fetch(:model).config.with(training: { steps: 2000, learning_rate: 0.0001, warmup_steps: 100, seed: 1337,
      eval_every: 100, checkpoint_every: 100, early_stopping_patience: 0, choice_microbatch: 4, gradient_accumulation: 8,
      balance_sources: false, balance_labels: false, paired_sampling: false, resample_negatives: false, track_coverage: true })
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: cfg)
    panels.each do |name, rows|
      EasyAI::Decision::Data::QualityAudit.integrity!(rows)
      rows.each do |row|
        collator.state_tokens(row.fetch("state"))
        row.fetch("options").each { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }
      end
    end
    train = panels.values_at("candidate-train.jsonl", "control-train.jsonl").flatten(1)
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(train, *panels.values_at("validation.jsonl", "calibration.jsonl", "test.jsonl", "natural-test.jsonl"))
    panels.values_at("validation.jsonl", "calibration.jsonl", "test.jsonl").flatten(1).each do |row|
      raise "Historical evaluation record" if historical.include?(DecisionFactual.material(row.fetch("state")))
    end
    # Public transfer and routing panels are explicitly historical regression, not acceptance.
    panels["regression.jsonl"] = raw("runs/decision/factual-v03/pilot/data/test.jsonl").reject { |row| row.fetch("source").start_with?("Factual-") }
    panels["v03-regression.jsonl"] = raw("runs/decision/factual-v03/pilot/data/test.jsonl").select { |row| row.fetch("source").start_with?("Factual-") }
    FileUtils.mkdir_p(File.join(root, "data"))
    panels.each { |name, rows| DecisionRobust.write_rows(File.join(root, "data", name), rows) }
    protocol = { "version" => 4, "created_at" => Time.now.utc.iso8601, "config" => cfg.to_h,
      "parent_sha256" => sha(File.join(PARENT, "weights.pt")), "review_sha256" => sha(review_path), "fresh_review_sha256" => sha(fresh_review_path), "historical_files_sha256" => history_files,
      "files_sha256" => panels.keys.to_h { |name| [name, sha(File.join(root, "data", name))] },
      "counts" => panels.transform_values { |rows| rows.group_by { |row| [row.fetch("source"), row.fetch("language"), row.fetch("options").size, row.fetch("target")].join("/") }.transform_values(&:size) },
      "sampling" => EasyAI::Decision::Data::JudgmentSampler::SIGNATURE, "steps" => 2000, "known_budget" => 1000,
      "single_fact_updates" => 200, "fit" => { "maximum_steps" => 1000, "accuracy" => 0.99, "mixed_groups" => 0.95, "stable_checks" => 2 },
      "mastery" => { "known_balanced" => 0.95, "mixed_groups" => 0.9, "dev_balanced" => 0.55, "stable_checks" => 2 },
      "publication" => { "binary_balanced" => 0.75, "joint_known_recall" => 0.7, "joint_unknown_recall" => 0.7, "mixed_groups" => 0.5, "parent_gain" => 0.05, "confirmation_seed" => 2027, "natural_tolerance" => 0.03, "natural_minimum_accuracy" => 0.65 },
      "natural_policy" => "Training uses only reviewed agreement-known BoolQ/OCNLI rows; disputes, neutral and DuReader excluded. Fresh acceptance uses a separate history-excluded 100-row blind review, retaining agreement/nonexcluded decisions. Single assistant reviewer, not independent human adjudication. Actual sample shortfall and class/domain counts disclosed; binary natural evaluation is explicitly gold-known only.",
      "natural_acceptance_available" => true,
      "selection" => "Binary and joint separately; worst-language class macro recall, then lower NLL. Natural-short-record evidence required for promotion. Seed 2027 only if eligible with five-point parent gain in both languages.",
      "scope" => "Own shared bilingual 6.63M encoder, CE, explicit record/claim. Canonical factual development and new combinations; public transfer/routing are regression only. No external weights, RL, business integration or automatic architecture fallback.",
      "deviations" => ["Natural qualification uses only reviewed agreement rows; small reviewed sample is disclosed rather than inferred eligibility from keywords.",
        "Canonical evaluation spans 100 dev/40 calibration/200 test bilingual semantic frames; held state/options views are transformations within these frames, never independent units.",
        "Control retains v03 base world distribution (128 families); reviewed candidate has 32 complete counterfactual lexical frames. Row counts/exposures are disclosed, not asserted equal independent coverage."] }
    write(File.join(root, "protocol.json"), protocol)
    puts JSON.pretty_generate(protocol.slice("counts", "natural_policy", "deviations"))
  end
end
