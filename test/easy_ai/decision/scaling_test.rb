require "test_helper"
require_relative "../../../benchmarks/decision/scaling"

# Coupled allocation and expansion invariants need joint assertions.
# rubocop:disable Minitest/MultipleAssertions
class ScalingTest < Minitest::Test
  Corpus = EasyAI::Decision::Data::ScalableJudgmentCorpus

  def test_dynamic_split_offsets_and_legacy_default_records
    corpus = EasyAI::Decision::Data::JudgmentCorpus

    assert_equal({ "train" => 0, "validation" => 64, "calibration" => 67, "test" => 72 }, corpus.split_offsets(train: 64, validation: 3, calibration: 5, test: 7))
    assert_raises(ArgumentError) { corpus.split_offsets(train: -1) }
    assert_raises(ArgumentError) { corpus.split_offsets("train" => 1, train: 2) }
    hashes = %w[5148b2aa82c8772c4ecda6112e4e9127b1deea677989416c04d0955e172b2f59 90fc261a017558200b5d753027f1b2679d69d199cc9acf60af66a6a480e89e94 61da224173e26897a8a2c409332363fe04a02924d25c4e351674feeb52c9997e 42978f2f0c4edaecb4e00a04ad81ad2859710aa6a60bdc759620123aea73ca21]
    %w[train validation calibration test].zip(hashes).each do |split, expected|
      rows = %w[en-US zh-CN].flat_map { |language| 2.times.flat_map { |index| corpus.new(split, index, language).rows } }

      assert_equal expected, Digest::SHA256.hexdigest(JSON.generate(rows))
    end
  end

  def test_scalable_worlds_have_independent_gold_complete_groups_and_disjoint_material
    counts = { train: 8, validation: 5, calibration: 3, test: 4 }
    panels = counts.to_h do |split, count|
      values = count.times.flat_map { |index| %w[en-US zh-CN].flat_map { |language| Corpus.new(split, index, language, counts: counts).rows } }
      [split, values]
    end

    assert panels.values.all? { |rows| EasyAI::Decision::Data::QualityAudit.integrity!(rows) }
    assert EasyAI::Decision::Data::QualityAudit.assert_disjoint!(*panels.values)
    assert_equal panels[:train].first(224), 1.times.flat_map { |index| %w[en-US zh-CN].flat_map { |language| Corpus.new(:train, index, language, counts: counts).rows } }
    patterns = panels[:train].select { |row| row.dig("world", "facts").size == 2 }.map { |row| row.dig("world", "row_truth_pattern").join }.uniq.sort

    assert_equal %w[falsefalse falsetrue truefalse truetrue], patterns
    groups = panels[:train].select { |row| row.dig("contrast_groups", "binding") }.group_by { |row| row.dig("contrast_groups", "binding") }

    assert groups.values.all? { |rows| rows.size == 2 }
    assert_raises(ArgumentError) { Corpus.new(:train, 8, "en-US", counts: counts) }
  end

  def test_actor_and_event_holdouts_reserve_actual_atoms
    counts = { train: 2, validation: 2 }
    make = ->(panel) { Corpus.new(:validation, 0, :"en-US", counts: counts, panel: panel).rows }
    canonical, actor, event = %w[canonical novel_actor novel_event].map { |panel| make.call(panel) }
    names = ->(rows) { rows.flat_map { |row| row.fetch("world").fetch("facts").map { |fact| fact.fetch("actor") } }.uniq }

    assert_empty names.call(canonical) & names.call(actor)
    assert event.flat_map { |row| row.fetch("world").fetch("facts").map { |fact| fact.fetch("event") } }.none? { |key| EasyAI::Decision::Data::FactualContrasts::TRAIN_EVENTS.include?(key) }
    assert EasyAI::Decision::Data::QualityAudit.assert_disjoint!(canonical, actor, event)
  end

  def test_new_actor_catalog_does_not_reuse_legacy_controlled_identities
    legacy = EasyAI::Decision::Data::RobustFacts
    old_en = legacy::EN_FIRST.product(legacy::EN_LAST).map { |a, b| "#{a} #{b}" }
    old_zh = legacy::ZH_FIRST.product(legacy::ZH_LAST).map(&:join)

    assert_empty old_en & (Corpus.actors("en-US", "canonical") + Corpus.actors("en-US", "novel_actor"))
    assert_empty old_zh & (Corpus.actors("zh-CN", "canonical") + Corpus.actors("zh-CN", "novel_actor"))
  end

  def test_capacity_expansion_preserves_outputs_and_new_units_learn
    Torch.manual_seed(1337)
    config = tiny_config(model: { position_encoding: :rotary, embedding_norm: true, score_mode: :matching, dropout: 0.0 })
    model = EasyAI::Decision::ChoiceModel.new(config).eval
    original = model.encoder.embedding.weight.detach.clone
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: config).call([example, example(id: "b", state: "blue", target: "b")])
    grown = EasyAI::Decision::Growth::Scale.apply(model, layers: 3, ffn_size: 64, seed: 2027).eval

    assert_tensor_close model.call(batch), grown.call(batch), 1e-5
    assert_operator grown.parameter_count, :>, model.parameter_count
    optimizer = EasyAI::Optim::AdamW.new(grown.named_parameters, learning_rate: 0.01)
    2.times do
      optimizer.zero_grad
      Torch::NN::Functional.cross_entropy(grown.call(batch), batch[:targets]).backward
      optimizer.step
    end

    assert_operator grown.encoder.blocks[1].attention.instance_variable_get(:@query).weight.grad.abs.sum.item, :>, 0
    assert_operator grown.encoder.blocks[0].ffn.up.weight.grad.narrow(0, 32, 32).abs.sum.item, :>, 0
    assert_tensor_close original, model.encoder.embedding.weight
    assert_raises(ArgumentError) { EasyAI::Decision::Growth::Scale.apply(model, layers: 0, ffn_size: 64, seed: 1) }
    assert_raises(ArgumentError) { EasyAI::Decision::Growth::Scale.apply(model, layers: 2, ffn_size: 16, seed: 1) }
  end

  def test_symbol_enum_values_and_capacity_names_are_normalized
    config = tiny_config(model: { position_encoding: :rotary }, training: { device: :cpu, precision: :float32 }, input: { truncation: :error })

    assert_equal "rotary", config[:model]["position_encoding"]
    assert_equal "cpu", config[:training]["device"]
    assert_equal [4, 768], DecisionScaling::PROFILES.fetch(:base.to_s)
  end

  def test_four_query_sampler_balances_axes_and_languages_and_resumes_exactly
    Dir.mktmpdir do |directory|
      raw = 2.times.flat_map { |index| %w[en-US zh-CN].flat_map { |language| Corpus.new(:train, index, language, counts: { train: 2 }).rows } }
      raw = DecisionScaling.binary_rows(raw)
      dataset = write_dataset(File.join(directory, "data.jsonl"), raw)
      config = tiny_config(model: { position_encoding: :rotary }, input: { state_max_tokens: 256, question_option_max_tokens: 256 },
        training: { steps: 2, choice_microbatch: 4, gradient_accumulation: 8 })
      Torch.manual_seed(1337)
      trainer = DecisionScaling::QuadTrainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new,
        dataset: dataset, output: File.join(directory, "first"))
      sampled = trainer.send(:sample_indices, 32, rng: Random.new(1)).map { |index| dataset[index] }

      assert_equal({ "no" => 16, "yes" => 16 }, sampled.group_by(&:target).transform_values(&:size))
      assert_equal [8, 8, 8, 8], sampled.group_by { |row| [row.language, row.source] }.values.map(&:size).sort
      assert sampled.each_slice(4).all? { |rows| rows.map(&:state).uniq.size == 1 && rows.map(&:group_id).uniq.size == 1 }
      trainer.train(steps: 1)
      resumed = DecisionScaling::QuadTrainer.resume(trainer.last_checkpoint, dataset: dataset, output: File.join(directory, "resumed"))
      trainer.train
      resumed.train

      assert trainer.model.state_dict.all? { |key, tensor| (tensor - resumed.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal 64, resumed.state.fetch("examples_seen")
      assert_equal DecisionScaling::QuadTrainer::SIGNATURE, resumed.state.fetch("scale_sampling")
    end
  end

  def test_extension_rule_distinguishes_undertraining_trend_and_plateau
    metric = lambda do |balanced, groups|
      { "worst_language_balanced" => balanced, "by_language" => %w[en-US zh-CN].to_h { |language| [language, { "actor_binding_all_correct" => groups, "event_binding_all_correct" => 1.0 }] } }
    end
    history = [{ "step" => 1000, "canonical" => metric.call(0.8, 0.4) }, { "step" => 1500, "canonical" => metric.call(0.8, 0.4) }]

    assert DecisionScaling.extension_needed?({ "known_mastery_passed" => false }, history)
    refute DecisionScaling.extension_needed?({ "known_mastery_passed" => true }, history)
    history.last["canonical"] = metric.call(0.82, 0.4)

    assert DecisionScaling.extension_needed?({ "known_mastery_passed" => true }, history)
    history.last["canonical"] = metric.call(0.8, 0.45)

    assert DecisionScaling.extension_needed?({ "known_mastery_passed" => true }, history)
  end

  def test_budget_counts_probe_and_replayed_updates
    Dir.mktmpdir do |root|
      paths = %w[preflight/base-micro4/training.jsonl arms/base-32-1337/choice/training.jsonl]
      paths.each do |relative|
        path = File.join(root, relative)
        FileUtils.mkdir_p(File.dirname(path))
        File.write(path, "{\"step\":1}\n{\"step\":1}\n")
      end

      assert_equal 4, DecisionScaling.update_ledger(root).values.sum
      assert_raises(RuntimeError) { DecisionScaling.enforce_budget!(root, DecisionScaling::UPDATE_LIMIT - 3) }
      assert_equal 4, DecisionScaling.enforce_budget!(root, 0)
    end
  end

  def test_controlled_eligibility_requires_each_language_and_each_development_slice
    names = %w[canonical seen_combinations novel_actor]
    metric = { "by_language" => %w[en-US zh-CN].to_h do |language|
      [language, { "balanced" => 0.8, "binding_groups" => 2, "binding_all_correct" => 0.75, "actor_binding_groups" => 1,
        "actor_binding_all_correct" => 0.5, "event_binding_groups" => 1, "event_binding_all_correct" => 1.0 }]
    end }
    values = names.to_h { |name| [name, Marshal.load(Marshal.dump(metric))] }
    reference = names.to_h { |name| [name, { "by_language" => %w[en-US zh-CN].to_h { |language| [language, { "balanced" => 0.5 }] } }] }
    protocol = { "required_development" => names, "publication" => { "mixed_groups" => 0.5, "binary_balanced" => 0.75, "parent_gain" => 0.05 } }

    assert DecisionScaling.controlled_eligible?(values, reference, protocol)
    values["novel_actor"]["by_language"]["zh-CN"]["actor_binding_all_correct"] = 0.0

    refute DecisionScaling.controlled_eligible?(values, reference, protocol)
    values["novel_actor"]["by_language"]["zh-CN"]["actor_binding_all_correct"] = 0.5
    reference["seen_combinations"]["by_language"]["en-US"]["balanced"] = 0.78

    refute DecisionScaling.controlled_eligible?(values, reference, protocol)
  end

  def test_trace_audit_rejects_cpu_runs_and_counts_replayed_updates
    Dir.mktmpdir do |root|
      path = File.join(root, "arms/base-32-1337/choice/training.jsonl")
      FileUtils.mkdir_p(File.dirname(path))
      row = { "step" => 1, "train_loss" => 0.4, "device" => "cuda" }
      File.write(path, [row, row].map { |item| JSON.generate(item) }.join("\n") + "\n")

      assert_equal 2, DecisionScaling.trace_audit(root).values.first.fetch("actual_updates")
      File.write(path, JSON.generate(row.merge("device" => "cpu")) + "\n")

      assert_raises(RuntimeError) { DecisionScaling.trace_audit(root) }
    end
  end

  def test_extension_decision_rejects_modified_initial_results
    Dir.mktmpdir do |root|
      screen = File.join(root, "arms/base-32-1337/screen.json")
      FileUtils.mkdir_p(File.dirname(screen))
      File.write(screen, "{}")
      path = File.join(root, "extension-selection.json")
      DecisionScaling.write(path, { "extend" => [], "screen_sha256" => { "base-32" => DecisionScaling.sha(screen) } })
      DecisionScaling.write(path + ".sha256.json", { "sha256" => DecisionScaling.sha(path) })

      assert_empty DecisionScaling.extension_decision(root).fetch("extend")
      File.write(screen, "{\"step\":1}")

      assert_raises(RuntimeError) { DecisionScaling.extension_decision(root) }
    end
  end

  def test_supervised_sampler_gates_unknowns_preserves_pairs_and_balances_natural_targets
    rows = %w[en-US zh-CN].flat_map { |language| 2.times.flat_map { |index| Corpus.new(:train, index, language, counts: { train: 2 }).rows } }
    %w[en-US zh-CN].each do |language|
      %w[yes no unknown].each do |target|
        [2, 3].each do |count|
          next if target == "unknown" && count == 2
          # Unequal pool sizes check that targets are chosen before examples.
          (target == "yes" ? 9 : 1).times do |index|
            rows << { "id" => "#{language}/#{target}/#{count}/#{index}", "language" => language, "source" => "OCNLI",
              "state" => "A natural record.", "question" => "Is the claim supported?", "target" => target,
              "options" => EasyAI::Decision::Data::JudgmentCorpus.options(language, binary: count == 2) }
          end
        end
      end
      rows << { "id" => "#{language}/routing", "language" => language, "source" => "MASSIVE-Scenario", "state" => "A request.", "question" => "Which domain?",
        "target" => "0", "options" => 18.times.map { |index| { "id" => index.to_s, "text" => "Domain #{index}" } } }
    end
    Dir.mktmpdir do |directory|
      path = File.join(directory, "train.jsonl")
      DecisionRobust.write_rows(path, rows)
      dataset = EasyAI::Decision::Data::Dataset.new(path)
      config = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 }, training: { steps: 1002, choice_microbatch: 4, gradient_accumulation: 8 })
      trainer = DecisionScaling::SupervisedTrainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new,
        dataset: dataset, output: File.join(directory, "choice"))
      trainer.train(steps: 1)
      trainer.state["step"] = 500
      sample = -> { trainer.send(:sample_indices, 32, rng: Random.new(1)).map { |index| dataset[index] } }
      known = sample.call

      assert_equal 16, known.count { |row| row.source.start_with?("Factual-") }
      assert known.select { |row| row.source.start_with?("Factual-") }.all? { |row| row.options.size == 2 && row.target != "unknown" }
      assert_equal({ "no" => 4, "yes" => 4 }, known.select { |row| row.source == "OCNLI" }.group_by(&:target).transform_values(&:size))
      trainer.state["supervised_known_passed"] = false
      trainer.state["supervised_mastery_checks"] = 1
      trainer.state["stop_reason"] = "Known retention failed by the fixed 500-update supervised ceiling"
      original_checkpoint = trainer.save_checkpoint
      fork = DecisionScaling::RetainedTrainer.resume(original_checkpoint, dataset: dataset, output: File.join(directory, "fork"), fork_from_original: true)
      retained = fork.send(:sample_indices, 32, rng: Random.new(1)).map { |index| dataset[index] }

      assert_equal 24, retained.count { |row| row.source.start_with?("Factual-") }
      assert_equal 4, retained.count { |row| row.source == "OCNLI" }
      assert_equal 4, retained.count { |row| row.source == "MASSIVE-Scenario" }
      assert_equal [16, 16], retained.group_by(&:language).values.map(&:size).sort
      refute fork.state.key?("stop_reason")
      assert trainer.state.key?("stop_reason")
      assert_equal 1, fork.state.fetch("retained_mastery_checks")
      assert trainer.model.state_dict.all? { |name, tensor| (tensor - fork.model.state_dict.fetch(name)).abs.max.item.zero? }
      first_moments = trainer.optimizer.state_dict.fetch("tensors")
      fork_moments = fork.optimizer.state_dict.fetch("tensors")

      refute_empty first_moments
      assert first_moments.all? { |name, tensor| (tensor - fork_moments.fetch(name)).abs.max.item.zero? }
      assert_equal trainer.optimizer.state_dict.except("tensors"), fork.optimizer.state_dict.except("tensors")
      assert_raises(ArgumentError) { DecisionScaling::RetainedTrainer.resume(original_checkpoint, dataset: dataset, output: File.join(directory, "invalid-fork")) }
      fork.state["step"] = 1000
      fork.state["stop_reason"] = "Corrective known retention failed by the fixed 1000-step ceiling"
      terminal = fork.save_checkpoint
      terminal_resume = DecisionScaling::RetainedTrainer.resume(terminal, dataset: dataset, output: File.join(directory, "terminal-resume"))

      assert_equal fork.state.fetch("stop_reason"), terminal_resume.state.fetch("stop_reason")
      assert_equal 1000, terminal_resume.state.fetch("step")
      assert_equal 500, trainer.state.fetch("step")
      trainer.state.delete("stop_reason")
      trainer.state["supervised_known_passed"] = true
      joint = sample.call
      controlled = joint.select { |row| row.source.start_with?("Factual-") }

      assert_equal 32, joint.size
      assert_equal 8, controlled.count { |row| row.target == "unknown" }
      assert_equal 12, controlled.count { |row| row.target != "unknown" }
      assert controlled.all? { |row| row.options.size == 3 }
      assert_equal 6, joint.count { |row| row.source == "MASSIVE-Scenario" }
      assert_equal %w[no unknown yes], joint.select { |row| row.source == "OCNLI" && row.language == "zh-CN" }.map(&:target).sort
      assert controlled.select { |row| row.target == "unknown" }.each_slice(2).all? { |pair| pair.map(&:state).uniq.size == 1 && pair.map(&:question).uniq.size == 2 }
      trainer.save_checkpoint
      restored = DecisionScaling::SupervisedTrainer.resume(trainer.last_checkpoint, dataset: dataset, output: File.join(directory, "restored"))

      assert restored.state.fetch("supervised_known_passed")
      assert_equal joint.map(&:id), restored.send(:sample_indices, 32, rng: Random.new(1)).map { |index| dataset[index].id }
      trainer.train(steps: 501)
      restored.train(steps: 501)

      assert trainer.model.state_dict.all? { |name, tensor| (tensor - restored.model.state_dict.fetch(name)).abs.max.item < 1e-7 }
      assert_equal 64, restored.state.fetch("examples_seen")
    end
  end

  def test_natural_review_uses_agreements_and_balances_binary_development_without_predictions
    rows = %w[yes yes yes no unknown].each_with_index.map do |target, index|
      { "id" => index.to_s, "source" => "OCNLI", "language" => "zh-CN", "state" => "Record #{index}", "question" => "Claim #{index}",
        "target" => target, "agreement" => true, "excluded" => false }
    end
    rows << rows.first.merge("id" => "disagreement", "agreement" => false)
    rows << rows.first.merge("id" => "excluded", "excluded" => true)
    audit = { "decisions" => rows }
    balanced = DecisionScaling.qualified_natural(audit, :binary, balanced: true)

    assert_equal %w[no yes], balanced.map { |row| row.fetch("target") }.sort
    assert_equal 2, balanced.size
    assert balanced.all? { |row| row.fetch("options").size == 2 }
    assert_equal 5, DecisionScaling.qualified_natural(audit, "joint").size
    assert_raises(ArgumentError) { DecisionScaling.qualified_natural(audit, :joint, balanced: true) }
  end

  def test_supervised_publication_requires_joint_unknown_recall_and_bilingual_natural_guards
    names = %w[canonical seen_combinations novel_actor]
    metric = { "nll" => 0.4, "worst_language_balanced" => 0.8, "by_language" => %w[en-US zh-CN].to_h do |language|
      [language, { "balanced" => 0.8, "binding_groups" => 2, "binding_all_correct" => 0.8,
        "actor_binding_groups" => 1, "actor_binding_all_correct" => 0.8, "event_binding_groups" => 1, "event_binding_all_correct" => 0.8,
        "recall" => { "yes" => 0.8, "no" => 0.8, "unknown" => 0.8 } }]
    end }
    value = { "controlled" => names.to_h { |name| [name, Marshal.load(Marshal.dump(metric))] },
      "natural" => { "by_language" => %w[en-US zh-CN].to_h { |language| [language, { "accuracy" => 0.8 }] } } }
    reference = Marshal.load(Marshal.dump(value))
    reference.fetch("controlled").each_value { |metrics| metrics.fetch("by_language").each_value { |cell| cell["balanced"] = 0.5 } }
    protocol = { "required_development" => names, "publication" => { "mixed_groups" => 0.5, "binary_balanced" => 0.75, "parent_gain" => 0.05,
      "joint_known_recall" => 0.7, "joint_unknown_recall" => 0.7, "natural_minimum_accuracy" => 0.65, "natural_tolerance" => 0.03 } }

    assert DecisionScaling.supervised_eligible?(value, reference, "joint", protocol)
    value["controlled"]["novel_actor"]["by_language"]["zh-CN"]["recall"]["unknown"] = 0.69

    refute DecisionScaling.supervised_eligible?(value, reference, "joint", protocol)
    value["controlled"]["novel_actor"]["by_language"]["zh-CN"]["recall"]["unknown"] = 0.8
    value["natural"]["by_language"]["zh-CN"]["accuracy"] = 0.64

    refute DecisionScaling.supervised_eligible?(value, reference, "binary", protocol)
    value["natural"]["by_language"]["zh-CN"]["accuracy"] = 0.7

    refute DecisionScaling.supervised_eligible?(value, reference, "binary", protocol)
  end

  def test_frozen_protocol_rejects_changed_data_and_protocol_without_loading_a_model
    Dir.mktmpdir do |root|
      FileUtils.mkdir_p(File.join(root, "data"))
      artifacts = ["data/panel.jsonl", "data-audit.json", "cross-round-audit.json", "preflight/result.json"]
      artifacts.each do |name|
        FileUtils.mkdir_p(File.dirname(File.join(root, name)))
        File.write(File.join(root, name), "{}\n")
      end
      digest = ->(name) { Digest::SHA256.file(File.join(root, name)).hexdigest }
      protocol = { "parent_sha256" => "frozen-parent", "files_sha256" => { "panel.jsonl" => digest.call("data/panel.jsonl") },
        "data_audit_sha256" => digest.call("data-audit.json"), "cross_round_audit_sha256" => digest.call("cross-round-audit.json"),
        "hardware" => { "results_sha256" => { "preflight/result.json" => digest.call("preflight/result.json") } } }
      DecisionScaling.write(File.join(root, "protocol.json"), protocol)
      DecisionScaling.write(File.join(root, "protocol.json.sha256.json"), { "sha256" => digest.call("protocol.json") })
      parent = File.join(DecisionScaling::PARENT, "weights.pt")
      hash_file = ->(path) { path == parent ? "frozen-parent" : Digest::SHA256.file(path).hexdigest }
      original_sha = DecisionScaling.method(:sha)
      DecisionScaling.define_singleton_method(:sha, &hash_file)
      begin
        assert_equal protocol, DecisionScaling.verify(root)
        File.write(File.join(root, "data/panel.jsonl"), "changed")

        assert_raises(RuntimeError) { DecisionScaling.verify(root) }
        File.write(File.join(root, "data/panel.jsonl"), "{}\n")
        File.write(File.join(root, "protocol.json"), "{}")

        assert_raises(RuntimeError) { DecisionScaling.verify(root) }
      ensure
        DecisionScaling.define_singleton_method(:sha, original_sha)
      end
    end
  end

  def test_padding_slots_include_state_and_every_candidate
    # Two state lengths (2, 5), two candidates per row (3/4 and 6/1):
    # real tokens = 2+5+3+4+6+1; padded slots = 2*5 + 2*2*6.
    counts = DecisionScaling.batch_token_slots([[2, [3, 4]], [5, [6, 1]]])

    assert_equal({ "unpadded" => 21, "padded" => 34 }, counts)
    assert_raises(ArgumentError) { DecisionScaling.batch_token_slots([[2, [3, 4]], [5, [6]]]) }
    assert_raises(ArgumentError) { DecisionScaling.batch_token_slots([]) }
  end

  def test_development_retention_amendment_checks_both_languages_and_stability
    original = { "known_balanced" => 0.95, "dev_balanced" => 0.55, "mixed_groups" => 0.9, "stable_checks" => 2 }
    rules = DecisionScaling.diagnostic_mastery_rules(original)
    metric = { "by_language" => %w[en-US zh-CN].to_h do |language|
      [language, { "balanced" => 0.95, "actor_binding_groups" => 64, "event_binding_groups" => 64,
        "actor_binding_all_correct" => 0.85, "event_binding_all_correct" => 0.85 }]
    end }
    row = { "training_probe" => metric, "canonical" => metric }
    history = [row.merge("step" => 250), row.merge("step" => 500)]

    assert_in_delta 0.9, original.fetch("mixed_groups")
    assert_nil DecisionScaling.diagnostic_gate_history(history, original).fetch("first_stable_step")
    assert_equal 500, DecisionScaling.diagnostic_gate_history(history, rules).fetch("first_stable_step")
    %w[en-US zh-CN].product(%w[actor event]).each do |language, axis|
      broken = Marshal.load(Marshal.dump(history))
      broken.last["training_probe"]["by_language"][language]["#{axis}_binding_all_correct"] = 0.84

      assert_nil DecisionScaling.diagnostic_gate_history(broken, rules).fetch("first_stable_step")
    end
    broken = Marshal.load(Marshal.dump(history))
    broken.last["training_probe"]["by_language"]["zh-CN"]["balanced"] = 0.94

    assert_nil DecisionScaling.diagnostic_gate_history(broken, rules).fetch("first_stable_step")
  end

  def test_actor_pair_metric_is_correctness_with_independent_counts_and_option_ids
    rows = %w[en-US zh-CN].flat_map do |language|
      Corpus.new(:validation, 0, language, counts: { train: 0, validation: 1 }, panel: :novel_actor).rows.select { |row| row.dig("world", "profile") == "binary" }
    end
    oracle = rows.map { |row| row.fetch("target") }
    opposite = oracle.map { |target| target == "yes" ? "no" : "yes" }
    constant = Array.new(rows.size, "yes")
    [oracle, opposite, constant].zip([1.0, 0.0, 0.0]).each do |predictions, expected|
      independent = DecisionScaling.independent_actor_pairs(rows, predictions)
      # Reverse candidate order to expose accidental index/ID interchange.
      reversed = rows.map { |row| row.merge("options" => row.fetch("options").reverse) }
      logits = reversed.each_index.map { |index| reversed[index].fetch("options").map { |option| option.fetch("id") == predictions[index] ? 5.0 : -5.0 } }
      official = DecisionJudgment.metrics(reversed, logits)
      %w[en-US zh-CN].each do |language|
        assert_in_delta expected, independent.fetch(language).fetch("all_correct")
        assert_in_delta expected, official.dig("by_language", language, "actor_binding_all_correct")
      end
    end
    counted = DecisionScaling.independent_actor_pairs(rows, constant)

    assert_in_delta 0.5, counted.fetch("en-US").fetch("row_accuracy")
    assert_equal 4, counted.fetch("en-US").fetch("one_correct_pairs")
    assert_equal 0, counted.fetch("en-US").fetch("zero_correct_pairs")
    assert_equal 0, counted.fetch("en-US").fetch("two_correct_pairs")
    exhaustive = %w[yes no].repeated_permutation(2).map do |choices|
      guesses = rows.map { |row| choices.fetch(row.dig("world", "query")) }
      DecisionScaling.independent_actor_pairs(rows, guesses).fetch("en-US").fetch("all_correct")
    end

    assert_in_delta 0.25, exhaustive.sum / exhaustive.size
    assert_raises(ArgumentError) { DecisionScaling.independent_actor_pairs(rows, oracle.drop(1)) }
    missing = rows.index { |row| row.dig("world", "mixed") }
    assert_raises(ArgumentError) { DecisionScaling.independent_actor_pairs(rows.reject.with_index { |_, index| index == missing }, oracle.reject.with_index { |_, index| index == missing }) }
  end

  def test_candidate_training_shuffle_keeps_target_id_and_cross_entropy_aligned
    rows = Corpus.new(:train, 0, "en-US", counts: { train: 1 }).rows.select { |row| row.dig("world", "profile") == "binary" }
    Dir.mktmpdir do |directory|
      dataset = write_dataset(File.join(directory, "data.jsonl"), rows)
      config = tiny_config(model: { dropout: 0.0 }, input: { state_max_tokens: 256, question_option_max_tokens: 256 })
      trainer = EasyAI::Decision::CandidateTrainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new,
        dataset: dataset, output: File.join(directory, "choice"))
      collator = trainer.instance_variable_get(:@collator)
      original_call = collator.method(:call)
      captured = []
      collator.define_singleton_method(:call) do |examples, **options|
        batch = original_call.call(examples, **options)
        captured << [examples, batch]
        batch
      end
      loss = trainer.send(:loss_for, dataset.first(16), seed: 41)
      examples, batch = captured.fetch(0)
      targets = batch.fetch(:targets).to_a

      assert_equal examples.map(&:target), targets.each_with_index.map { |target, index| examples[index].options.fetch(target).fetch("id") }
      assert examples.each_with_index.any? { |example, index| example.options != dataset[index].options }
      scores = trainer.model.call(batch).to_a
      expected = scores.each_with_index.sum do |values, index|
        maximum = values.max
        Math.log(values.sum { |value| Math.exp(value - maximum) }) + maximum - values.fetch(targets[index])
      end / scores.size

      assert_in_delta expected, loss.item, 1e-5
      assert rows.all? { |row| row.fetch("options").map { |option| option.fetch("id") } == %w[yes no] }
    end
  end
end
