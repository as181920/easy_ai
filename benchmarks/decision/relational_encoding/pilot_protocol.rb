module DecisionRelational
  module_function

  def natural_rows(audit, profile)
    JSON.parse(File.read(audit)).fetch("decisions").select { |row| row.fetch("agreement") && !row.fetch("excluded") && (profile == "joint" || row.fetch("target") != "unknown") }.map do |row|
      row.slice("id", "group_id", "language", "state", "question", "target", "source").merge(
        "id" => "#{row.fetch('id')}:#{profile}", "profile" => profile,
        "options" => EasyAI::Decision::Data::JudgmentCorpus.options(row.fetch("language"), binary: profile == "binary"))
    end
  end

  def confirmed_fit!(root, selection: "default")
    choice = JSON.parse(File.read(fit_selection_path(root, selection)))
    winner = choice.fetch("winner")
    raise "No qualified fitting recipe" unless winner
    check_confirmation!(root, winner, selection: selection)
    result = JSON.parse(File.read(File.join(root, "fit-#{winner}-2027/result.json")))
    raise "Second-seed fitting confirmation failed" unless result.fetch("passed")
    winner
  end

  def prepare_pilot(root, variant: "standard")
    protocol = verify(root)
    selection = variant.to_s == "focused-candidate" ? "candidate" : "default"
    winner = confirmed_fit!(root, selection: selection)
    directory = pilot_directory(root, variant)
    unless Dir.glob(File.join(root, "pilot*/acceptance-opened.json")).empty?
      raise "Acceptance was observed; reserve new panels before another pilot"
    end
    authorize_focused!(root) if variant.to_s == "focused"
    authorize_pilot_failure!(root, "focused") if variant.to_s == "focused-candidate"
    raise "Pilot protocol exists" if File.exist?(directory)
    raw_config, initialization = recipe_config(root, protocol, winner)
    config = EasyAI::Decision::Config.new(raw_config).with(model: { dropout: 0.1 }, training: {
      steps: 2000, learning_rate: 0.0001, warmup_steps: 100
    })
    if focused_variant?(variant)
      config = config.with(model: { dropout: 0.0 }, training: { steps: 3000, learning_rate: 0.0003, warmup_steps: 0, choice_microbatch: 16, gradient_accumulation: 2 })
    end
    old = DecisionJudgment.verify(SOURCE)
    raise "v0.4 did not stop at fitting" if JSON.parse(File.read(File.join(SOURCE, "fit/result.json"))).fetch("passed")
    forbidden = %w[evaluation.json acceptance-opened.json calibration.json]
    raise "v0.4 acceptance already observed" if forbidden.any? { |name| File.exist?(File.join(SOURCE, name)) }
    raise "Unexpected v0.4 acceptance predictions" unless Dir.glob(File.join(SOURCE, "predictions-*.jsonl")).empty?
    panels = %w[candidate-train.jsonl candidate-probe.jsonl validation.jsonl].to_h { |name| [name, rows(File.join(root, "data", name))] }
    %w[calibration.jsonl test.jsonl natural-test.jsonl regression.jsonl].each { |name| panels[name] = rows(File.join(SOURCE, "data", name)) }
    review = File.join(root, "review-development/audit.json")
    panels["natural-development.jsonl"] = %w[binary joint].flat_map { |profile| natural_rows(review, profile) }
    train = panels.fetch("candidate-train.jsonl")
    development = panels.fetch("validation.jsonl") + panels.fetch("natural-development.jsonl")
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(train, development, panels.fetch("calibration.jsonl"), panels.fetch("test.jsonl") + panels.fetch("natural-test.jsonl"))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Decision::Checkpoint.load(PARENT).fetch(:tokenizer), config: config)
    panels.each_value do |values|
      EasyAI::Decision::Data::QualityAudit.integrity!(values)
      values.each do |row|
        collator.state_tokens(row.fetch("state"))
        row.fetch("options").each { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")) }
      end
    end
    FileUtils.mkdir_p(File.join(directory, "data"))
    panels.each { |name, values| DecisionRobust.write_rows(File.join(directory, "data", name), values) }
    rules = old.slice("mastery", "publication", "natural_policy")
    rules["mastery"]["mixed_groups"] = 0.9
    value = rules.merge("variant" => variant.to_s, "known_budget" => focused_variant?(variant) ? 2000 : 1000,
      "winner" => winner, "initialization" => initialization, "config" => config.to_h,
      "created_at" => Time.now.utc.iso8601, "fit_selection_file" => File.basename(fit_selection_path(root, selection)), "fit_selection_sha256" => sha(fit_selection_path(root, selection)),
      "confirmation_sha256" => sha(File.join(root, "fit-#{winner}-2027/result.json")),
      "parent_weights_sha256" => protocol.fetch("parent_weights_sha256"), "review_sha256" => sha(review),
      "files_sha256" => panels.keys.to_h { |name| [name, sha(File.join(directory, "data", name))] },
      "natural_training_counts" => train.reject { |row| row.fetch("source").start_with?("Factual-") || row.fetch("source") == "MASSIVE-Scenario" }.group_by { |row| row.fetch("language") }.transform_values(&:size),
      "acceptance_exposure_audit" => "v0.4 fitting failed; its frozen runner skips pilot/calibration/acceptance on failure, and no acceptance result/marker/predictions exist. v0.5 fitting drivers read only fit and validation. Review/integrity qualification before freeze is disclosed; source v0.4 test data stay reserved, development remains observed.",
      "sampling" => { "base" => EasyAI::Decision::Data::JudgmentSampler::SIGNATURE, "focused" => focused_variant?(variant) ? FocusedTrainer::SIGNATURE : nil },
      "source_sha256" => Dir.glob("{lib/easy_ai,benchmarks/decision}/**/*.rb").sort.to_h { |path| [path, sha(path)] })
    if focused_variant?(variant)
      reference = pilot_protocol(root)
      %w[validation.jsonl natural-development.jsonl].each do |name|
        raise "Baseline data differs" unless value.fetch("files_sha256").fetch(name) == reference.fetch("files_sha256").fetch(name)
      end
      baseline = File.join(pilot_directory(root, "standard"), "baseline.json")
      expected = JSON.parse(File.read(File.join(pilot_directory(root, "standard"), "baseline-sha256.json"))).fetch("sha256")
      raise "Common-parent baseline changed" unless sha(baseline) == expected
      FileUtils.cp(baseline, File.join(directory, "baseline.json"))
      write(File.join(directory, "baseline-sha256.json"), { "sha256" => expected })
      value["baseline_reused_sha256"] = expected
    end
    value.fetch("source_sha256").each do |path, digest|
      raise "Source changed during pilot freeze" unless sha(path) == digest
      destination = File.join(directory, "source-frozen", path)
      FileUtils.mkdir_p(File.dirname(destination))
      FileUtils.cp(path, destination)
    end
    write(File.join(directory, "protocol.json"), value)
    write(File.join(directory, "protocol-sha256.json"), { "sha256" => sha(File.join(directory, "protocol.json")) })
    puts "Frozen supervised pilot for #{winner}; natural training shortfall: #{value.fetch('natural_training_counts')} versus goal 200/language"
  end

  def pilot_protocol(root, variant: "standard")
    verify(root)
    directory = pilot_directory(root, variant)
    path = File.join(directory, "protocol.json")
    raise "Pilot protocol changed" unless sha(path) == JSON.parse(File.read(File.join(directory, "protocol-sha256.json"))).fetch("sha256")
    protocol = JSON.parse(File.read(path))
    raise "Pilot variant mismatch" unless protocol.fetch("variant", "standard") == variant.to_s
    selection_file = protocol.fetch("fit_selection_file", "fit-selection.json")
    unless %w[fit-selection.json fit-selection-candidate.json].include?(selection_file) &&
        sha(File.join(root, selection_file)) == protocol.fetch("fit_selection_sha256")
      raise "Pilot fitting selection changed"
    end
    protocol.fetch("files_sha256").each do |name, digest|
      raise "Pilot data changed" unless sha(File.join(directory, "data", name)) == digest
    end
    protocol
  end
end
