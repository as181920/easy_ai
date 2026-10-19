module DecisionRelational
  module_function

  def extra_event_development
    events = EasyAI::Decision::Data::FactualContrasts::EVENTS.keys - EasyAI::Decision::Data::FactualContrasts::TRAIN_EVENTS
    10.times.flat_map do |index|
      %w[en-US zh-CN].flat_map do |language|
        generator = EasyAI::Decision::Data::JudgmentCorpus.new("validation", 1000 + index, language)
        generator.instance_variable_set(:@group, "relational-v05:development-extra:#{index}")
        generator.instance_variable_set(:@events, events.rotate(index % events.size).first(2))
        generator.rows.select { |row| row.dig("world", "profile") == "binary" }
      end
    end
  end

  def composition(root, arm, seed)
    verify(root)
    directory = File.join(root, "fit-#{arm}-#{seed}")
    result = JSON.parse(File.read(File.join(directory, "result.json")))
    loaded = EasyAI::Decision::Checkpoint.load(result.fetch("checkpoint"))
    raw = DecisionJudgment.panel(root, "validation", "binary")
    views = acceptance_views(raw)
    views["irrelevant_note"] = raw.map { |row| row.merge("state" => row.fetch("state") + (row.fetch("language") == "zh-CN" ? "日志已保存。" : " The log is stored.")) }
    extra = extra_event_development
    EasyAI::Decision::Data::QualityAudit.integrity!(extra)
    train = rows(File.join(root, "data/candidate-train.jsonl"))
    EasyAI::Decision::Data::QualityAudit.assert_disjoint!(train, extra)
    data_path = File.join(root, "diagnostics/data/new-events.jsonl")
    FileUtils.mkdir_p(File.dirname(data_path))
    DecisionRobust.write_rows(data_path, extra) unless File.exist?(data_path)
    raise "Extra development data changed" unless rows(data_path) == extra
    views["events_absent_from_controlled_training"] = extra
    actors = train.filter_map { |row| row["world"] }.flat_map { |world| world.fetch("facts").map { |fact| fact.fetch("actor") } }.to_set
    novel = raw.group_by { |row| [row.fetch("language"), row.fetch("group_id")] }.values.select do |family|
      family.first.fetch("world").fetch("facts").all? { |fact| !actors.include?(fact.fetch("actor")) }
    end.flatten
    views["new_actor_identities"] = novel unless novel.empty?
    device = DecisionRobust.device
    values = views.to_h do |name, panel|
      value = DecisionJudgment.measure(loaded.fetch(:model), loaded.fetch(:tokenizer), panel, device)
      puts "Composition #{arm}/#{seed} #{name}: #{value.fetch('by_language').transform_values { |cell| cell.slice('balanced', 'actor_binding_all_correct', 'event_binding_all_correct') }}"
      $stdout.flush
      [name, value]
    end
    model = loaded.fetch(:model)
    model.encoder.instance_variable_set(:@rotary, false)
    model.encoder.instance_variable_set(:@position_scale, 0.0)
    model.encoder.blocks.each { |block| block.attention.instance_variable_set(:@rotary, false) }
    fitting = rows(File.join(root, "data/fit.jsonl"))
    values["positions_disabled_training_diagnostic"] = DecisionJudgment.measure(model, loaded.fetch(:tokenizer), fitting, device)
    model.to("cpu")
    write(File.join(directory, "composition.json"), { "views" => values, "checkpoint_weights_sha256" => loaded.fetch(:weights_fingerprint),
      "extra_development_sha256" => sha(data_path),
      "scope" => "Supplementary observed development after recipe selection; no acceptance, no training. New events mean absent from this controlled training split, not necessarily unseen in the own parent. Extra-note records exceed the one/two-proposition contract. Removing positions is an out-of-distribution mechanism diagnostic, not a trained competing recipe." })
  end
end
