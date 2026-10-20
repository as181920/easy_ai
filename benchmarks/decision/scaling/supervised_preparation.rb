module DecisionScaling
  module_function

  def qualified_natural(audit, profile, balanced: false)
    profile = profile.to_s
    raise ArgumentError, "Unknown natural judgment profile" unless %w[binary joint].include?(profile)
    rows = audit.fetch("decisions").select { |row| row.fetch("agreement") && !row.fetch("excluded") && %w[BoolQ OCNLI].include?(row.fetch("source")) }
    rows = rows.reject { |row| row.fetch("target") == "unknown" } if profile == "binary"
    if balanced
      raise ArgumentError, "Only binary natural balancing is defined" unless profile == "binary"
      rows = rows.group_by { |row| row.fetch("language") }.values.flat_map do |items|
        buckets = %w[yes no].map { |label| items.select { |row| row.fetch("target") == label }.sort_by { |row| sha_text("natural-dev:#{row.fetch('id')}") } }
        count = buckets.map(&:size).min
        raise "No balanced natural development" unless count.positive?
        buckets.flat_map { |bucket| bucket.first(count) }
      end
    end
    rows.map do |row|
      row.slice("id", "group_id", "language", "state", "question", "target", "source").merge("id" => "#{row.fetch('id')}:#{profile}", "profile" => profile,
        "options" => EasyAI::Decision::Data::JudgmentCorpus.options(row.fetch("language"), binary: profile == "binary"))
    end
  end

  def sha_text(value)
    Digest::SHA256.hexdigest(value)
  end

  def prepare_supervised(root)
    protocol = verify(root)
    path = File.join(root, "supervised-protocol.json")
    raise "Supervised protocol already frozen" if File.exist?(path)
    choice_path = File.join(root, "selection.json")
    raise "Controlled selection changed" unless sha(choice_path) == JSON.parse(File.read(choice_path + ".sha256.json")).fetch("sha256")
    choice = JSON.parse(File.read(choice_path))
    raise "No known-mastery-qualified controlled selection" unless choice.fetch("cell")
    selected_profile, scale = choice.fetch("cell").split("-")
    scale = Integer(scale)
    profiles = ["base", selected_profile].uniq
    inputs = %w[train development].to_h do |name|
      file = File.join(root, "natural-review", "#{name}-audit.json")
      raise "Natural review changed" unless sha(file) == JSON.parse(File.read(file + ".sha256.json")).fetch("sha256")
      [name, file]
    end
    old_review = "runs/decision/judgment-v04/review/audit.json"
    regression_review = "runs/decision/relational-v05/review-development/audit.json"
    replay_path = "runs/decision/relational-v05/pilot/data/candidate-train.jsonl"
    fresh = JSON.parse(File.read(inputs.fetch("train")))
    old = JSON.parse(File.read(old_review))
    development = JSON.parse(File.read(inputs.fetch("development")))
    controlled = DecisionRelational.rows(File.join(root, "data/train.jsonl")).select { |row| row.fetch("group_id").split(":").last.to_i < scale }
    natural = %w[binary joint].flat_map { |profile| qualified_natural(fresh, profile) + qualified_natural(old, profile) }
    routing = DecisionRelational.rows(replay_path).select { |row| row.fetch("source") == "MASSIVE-Scenario" }
    panels = { "train" => controlled + natural + routing }
    %w[binary joint].each do |profile|
      panels["natural-development-#{profile}"] = qualified_natural(development, profile, balanced: profile == "binary")
      panels["natural-regression-#{profile}"] = qualified_natural(JSON.parse(File.read(regression_review)), profile)
    end
    EasyAI::Decision::Data::QualityAudit.integrity!(panels.fetch("train"))
    %w[binary joint].each do |profile|
      EasyAI::Decision::Data::QualityAudit.assert_disjoint!(panels.fetch("train"), panels.fetch("natural-development-#{profile}"))
      raw = panels.fetch("natural-development-#{profile}")
      raise "Missing natural development language" unless raw.map { |row| row.fetch("language") }.uniq.sort == %w[en-US zh-CN]
    end
    controlled_dev = (protocol.fetch("required_development") + protocol.fetch("diagnostic_development")).flat_map { |name| panel(root, name, profile: "joint") }
    reserved = %w[calibration test calibration-novel_actor test-novel_actor calibration-novel_event test-novel_event].map { |name| DecisionRelational.rows(File.join(root, "data", "#{name}.jsonl")) }
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(panels.fetch("train"), controlled_dev, *reserved)
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: loaded.fetch(:tokenizer), config: loaded.fetch(:model).config.with(input: { truncation: "error" }))
    panels.each_value do |rows|
      rows.each do |row|
        collator.state_tokens(row.fetch("state"))
        row.fetch("options").each { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }
      end
    end
    directory = File.join(root, "supervised-data")
    FileUtils.mkdir_p(directory)
    panels.each { |name, rows| DecisionRobust.write_rows(File.join(directory, "#{name}.jsonl"), rows) }
    decision = extension_decision(root)
    checkpoints = profiles.to_h do |profile|
      cell = "#{profile}-#{scale}"
      suffix = decision.fetch("extend").include?(cell) ? "extended" : "screen"
      result_path = File.join(root, "arms", "#{cell}-1337", "#{suffix}.json")
      result = JSON.parse(File.read(result_path))
      raise "Controlled endpoint changed" unless sha(result_path) == choice.fetch("results_sha256").fetch(cell)
      [profile, { "checkpoint" => result.fetch("checkpoint"), "steps" => result.fetch("step"), "weights_sha256" => sha(File.join(result.fetch("checkpoint"), "weights.pt")) }]
    end
    configs = profiles.to_h do |profile|
      config = EasyAI::Decision::Config.new(protocol.fetch("config_by_profile").fetch(profile)).with(training: {
        steps: 3000, choice_microbatch: protocol.dig("hardware", "full_microbatch", profile),
        gradient_accumulation: 32 / protocol.dig("hardware", "full_microbatch", profile)
      })
      [profile, config.to_h]
    end
    sources = Dir["{lib/easy_ai,benchmarks/decision}/**/*.rb"].sort
    sources.each do |file|
      target = File.join(root, "supervised-source-frozen", file)
      FileUtils.mkdir_p(File.dirname(target))
      FileUtils.cp(file, target)
    end
    value = { "selection_sha256" => sha(choice_path), "scale" => scale, "profiles" => profiles, "steps" => 3000,
      "initial_checkpoints" => checkpoints, "config_by_profile" => configs, "sampler" => SupervisedTrainer::SIGNATURE,
      "files_sha256" => panels.keys.to_h { |name| ["#{name}.jsonl", sha(File.join(directory, "#{name}.jsonl"))] },
      "review_sha256" => inputs.merge("prior_training" => old_review, "prior_regression" => regression_review).to_h { |_name, file| [file, sha(file)] },
      "replay_source_sha256" => sha(replay_path), "source_sha256" => sources.to_h { |file| [file, sha(file)] },
      "plan_sha256" => sha("docs/decision/v06-supervised-plan.md"), "truncations" => collator.truncated,
      "qualified_new_training" => fresh.fetch("qualified_counts"),
      "combined_natural_training" => natural.select { |row| row.fetch("profile") == "joint" }.group_by { |row| "#{row.fetch('language')}/#{row.fetch('target')}" }.transform_values(&:size),
      "natural_development_counts" => %w[binary joint].to_h { |profile| [profile, panels.fetch("natural-development-#{profile}").group_by { |row| "#{row.fetch('language')}/#{row.fetch('target')}" }.transform_values(&:size)] },
      "scope" => "Conditional supervised intervention: natural/routing retention then unknown exposure. Fresh binary natural development is label-balanced before predictions. Prior natural development is regression only. No acceptance opened." }
    FileUtils.cp("docs/decision/v06-supervised-plan.md", File.join(root, "supervised-plan-frozen.md"))
    write(path, value)
    write(path + ".sha256.json", { "sha256" => sha(path) })
    puts "Frozen supervised scale recipe: #{profiles}/#{scale}; #{fresh.fetch('qualified_counts')}"
  end

  def supervised_protocol(root)
    verify(root)
    path = File.join(root, "supervised-protocol.json")
    raise "Supervised protocol changed" unless sha(path) == JSON.parse(File.read(path + ".sha256.json")).fetch("sha256")
    value = JSON.parse(File.read(path))
    raise "Controlled selection changed" unless sha(File.join(root, "selection.json")) == value.fetch("selection_sha256")
    value.fetch("files_sha256").each { |name, digest| raise "Supervised data changed" unless sha(File.join(root, "supervised-data", name)) == digest }
    value.fetch("review_sha256").each { |file, digest| raise "Natural review changed" unless sha(file) == digest }
    value.fetch("initial_checkpoints").each_value { |cell| raise "Initial controlled weights changed" unless sha(File.join(cell.fetch("checkpoint"), "weights.pt")) == cell.fetch("weights_sha256") }
    value
  end
end
