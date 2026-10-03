require "test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/judgment"

# Coupled dataset/curriculum invariants are checked together to expose their interaction.
# rubocop:disable Minitest/MultipleAssertions
class JudgmentTest < Minitest::Test
  def generated(split = "train")
    %w[en-US zh-CN].flat_map { |language| [0, 1].flat_map { |index| EasyAI::Decision::Data::JudgmentCorpus.new(split, index, language).rows } }
  end

  def mixture
    generated + %w[en-US zh-CN].flat_map do |language|
      row = generated.find { |item| item.fetch("language") == language && item.fetch("options").size == 2 }
      options = [{ "id" => "yes", "text" => "routing" }] + (1..17).map { |index| { "id" => index.to_s, "text" => "domain #{index}" } }
      [row.merge("id" => "public:#{language}", "source" => "BoolQ"), row.merge("id" => "routing:#{language}", "source" => "MASSIVE-Scenario", "options" => options, "target" => "yes")]
    end
  end

  def test_independent_fact_flips_actual_metadata_and_rendered_gold
    rows = generated

    assert EasyAI::Decision::Data::QualityAudit.integrity!(rows)
    frames = rows.select { |row| row.dig("world", "profile") == "joint" }.group_by { |row| [row.fetch("group_id"), row.fetch("language")] }

    assert frames.values.all? { |items| items.map { |row| row.dig("world", "row_truth_pattern") }.uniq.size == 4 }
    rows.each do |row|
      world = row.fetch("world")
      world.fetch("facts").each do |fact|
        phrase = EasyAI::Decision::Data::FactualContrasts::EVENTS.fetch(fact.fetch("event")).fetch((row.fetch("language") == "zh-CN" ? 0 : 2) + (fact.fetch("truth") ? 0 : 1))

        assert_includes row.fetch("state"), fact.fetch("actor")
        assert_includes row.fetch("state"), phrase
      end
    end

    assert_equal %w[absent_actor known missing_event], rows.map { |row| row.dig("world", "kind") }.uniq.sort
    %w[binding fact_flip question_flip].each do |axis|
      groups = rows.select { |row| row.dig("contrast_groups", axis) }.group_by { |row| row.dig("contrast_groups", axis) }

      assert(groups.values.all? { |items| items.size == 2 }, "Incomplete #{axis} group")
    end
  end

  def test_audit_rejects_bad_gold_metadata_duplicates_and_leakage
    rows = generated
    wrong = Marshal.load(Marshal.dump(rows))
    wrong.first["target"] = wrong.first.fetch("target") == "yes" ? "no" : "yes"
    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.integrity!(wrong) }
    wrong = Marshal.load(Marshal.dump(rows))
    wrong.first.fetch("world")["row_truth_pattern"] = [true, true]
    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.integrity!(wrong) }
    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.integrity!(rows + [rows.first]) }
    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.assert_disjoint!(rows, [rows.first.merge("group_id" => "other")]) }
    assert EasyAI::Decision::Data::QualityAudit.assert_disjoint!(rows, generated("test"))
  end

  def test_blind_export_and_exact_review_coverage
    blind, selected = EasyAI::Decision::Data::QualityAudit.export(generated, per_source: 8)

    assert_equal 8, blind.size
    assert blind.none? { |row| row.key?("target") || row.key?("world") }
    reviews = selected.map { |row| { "review_id" => row.fetch("review_id"), "target" => row.fetch("target"), "reason" => "Explicit facts agree." } }

    assert EasyAI::Decision::Data::QualityAudit.adjudicate(selected, reviews).fetch("by_source").values.all? { |cell| cell.fetch("agreement") == 1 }
    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.adjudicate(selected, reviews.drop(1)) }
    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.adjudicate(selected, reviews + [reviews.first]) }
  end

  def test_audit_detects_corrupted_rendering_not_only_metadata
    row = generated.first
    changed = row.merge("state" => "An unrelated record.")

    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.integrity!([changed]) }
    changed = row.merge("question" => "A different actor did something.")

    assert_raises(ArgumentError) { EasyAI::Decision::Data::QualityAudit.integrity!([changed]) }
  end

  def test_legacy_audit_distinguishes_gold_errors_from_stale_reporting_metadata
    rows = EasyAI::Decision::Data::RobustFacts.new("transport", 0, "en-US").rows
    value = EasyAI::Decision::Data::QualityAudit.legacy_integrity_report(rows)

    assert_equal 0, value.fetch("gold_mismatches")
    assert_equal 2, value.fetch("stale_truth_pattern")
    changed = rows.map(&:dup)
    changed.first["target"] = "unknown"

    assert_equal 1, EasyAI::Decision::Data::QualityAudit.legacy_integrity_report(changed).fetch("gold_mismatches")
  end

  def test_acceptance_cannot_open_from_unfinished_pilots
    Dir.mktmpdir do |directory|
      DecisionJudgment.stubs(:verify).returns({})
      File.write(File.join(directory, "confirmation.json"), JSON.generate("confirmation_required" => false))
      FileUtils.mkdir_p(File.join(directory, "fit"))
      File.write(File.join(directory, "fit/result.json"), JSON.generate("passed" => true))

      assert_raises(RuntimeError) { DecisionJudgment.freeze_policy(directory) }
      refute_path_exists File.join(directory, "acceptance-opened.json")
    end
  end

  def test_exact_mixture_classes_languages_and_deterministic_stages
    rows = mixture.map { |row| EasyAI::Decision::Data::Example.new(row) }
    sampler = EasyAI::Decision::Data::JudgmentSampler.new(rows)
    %w[known joint].each do |stage|
      cells = Hash.new(0)
      5.times do |step|
        indexes = sampler.sample(32, rng: Random.new(step), step: step, stage: stage)

        assert_equal 32, indexes.size
        assert_equal indexes, sampler.sample(32, rng: Random.new(step), step: step, stage: stage)
        indexes.each do |i|
          row = rows[i]
          kind = row.source.start_with?("Factual-") ? "core" : row.source == "BoolQ" ? "natural" : "routing"
          cells[[kind, row.language, row.target]] += 1
          refute_equal "unknown", row.target if stage == "known"
        end
      end

      assert_equal 96, cells.select { |(kind, _, _), _| kind == "core" }.values.sum
      %w[natural routing].each { |kind| assert_equal 32, cells.select { |(key, _, _), _| key == kind }.values.sum }
      %w[en-US zh-CN].each do |language|
        assert_equal 48, cells.select { |(kind, lang, _), _| kind == "core" && lang == language }.values.sum
        assert_equal 16, cells[["core", language, "unknown"]] if stage == "joint"
      end
    end
    assert_raises(ArgumentError) { EasyAI::Decision::Data::JudgmentSampler.new(rows.drop(1)) }
  end

  def test_class_macro_and_mixed_groups_reject_constant_unknown_and_actor_blind
    rows = generated("test").select { |row| row.dig("world", "profile") == "joint" }
    perfect = rows.map { |row| row.fetch("options").map { |option| option.fetch("id") == row.fetch("target") ? 8.0 : 0.0 } }
    value = DecisionJudgment.metrics(rows, perfect)

    assert_in_delta(1.0, value.fetch("worst_language_balanced"))
    assert value.fetch("by_language").values.all? { |cell| cell.fetch("binding_all_correct") == 1.0 }
    unknown = DecisionJudgment.metrics(rows, Array.new(rows.size) { [0.0, 0.0, 8.0] })

    assert_in_delta 1.0 / 3, unknown.fetch("worst_language_balanced")
    gates = { "mixed_groups" => 0.5, "joint_known_recall" => 0.7, "joint_unknown_recall" => 0.7 }

    refute DecisionJudgment.eligible?(unknown, "joint", gates)
    assert DecisionJudgment.eligible?(value, "joint", gates)
  end

  def test_sparse_valid_buckets_still_produce_full_effective_batches
    rows = mixture.map { |row| EasyAI::Decision::Data::Example.new(row) }
    core, other = rows.partition { |row| row.source.start_with?("Factual-") }
    pairs = core.group_by { |row| row.contrast_groups.fetch("question_flip") }.values
    minimal = pairs.group_by do |pair|
      row = pair.first
      [row.language, row.options.size, row.source.end_with?("Single"), row.target == "unknown"]
    end.values.map(&:first).flatten
    sampler = EasyAI::Decision::Data::JudgmentSampler.new(minimal + other)

    assert_equal 32, sampler.sample(32, rng: Random.new(1), step: 0, stage: "joint").size
    assert_equal 32, sampler.sample(32, rng: Random.new(1), step: 4, stage: "joint").size
  end

  def test_failed_fitting_procedure_does_not_run_pilots_or_open_acceptance
    Dir.mktmpdir do |directory|
      FileUtils.mkdir_p(File.join(directory, "fit"))
      File.write(File.join(directory, "fit/result.json"), JSON.generate("passed" => false))
      %w[baseline fit report].each { |phase| DecisionJudgment.expects(:subprocess).with(directory, phase) }
      DecisionJudgment.expects(:subprocess).with(directory, "confirm").never
      DecisionJudgment.expects(:subprocess).with(directory, "evaluate").never
      DecisionJudgment.run_prepared(directory)

      refute_path_exists File.join(directory, "acceptance-opened.json")
    end
  end

  def test_resume_across_stage_boundary_preserves_parameters_optimizer_and_coverage
    Dir.mktmpdir do |directory|
      dataset = write_dataset(File.join(directory, "data.jsonl"), mixture)
      cfg = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 },
        training: { choice_microbatch: 4, gradient_accumulation: 8, track_coverage: true, steps: 2 })
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      Torch.manual_seed(1337)
      first = EasyAI::Decision::ChoiceModel.new(cfg)
      other = EasyAI::Decision::ChoiceModel.new(cfg)
      other.load_state_dict(first.state_dict)
      trainers = [first, other].each_with_index.map { |model, index| EasyAI::Decision::JudgmentTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, index.to_s)) }
      trainers.each { |trainer| trainer.train(steps: 1); trainer.advance!(mastered: true) }

      refute_equal 0, first.encoder.embedding.weight.grad.abs.sum.item
      resumed = EasyAI::Decision::JudgmentTrainer.resume(trainers.last.last_checkpoint, dataset: dataset, output: File.join(directory, "resume"))

      assert_equal "joint", resumed.state.fetch("judgment_stage")
      before = first.named_parameters.transform_values(&:object_id)
      rows = dataset.first(4)
      SemanticCoverageEvaluation.new(model: first, tokenizer: tokenizer, device: "cpu", batch_size: 2).collect(rows)

      assert_equal before, first.named_parameters.transform_values(&:object_id)
      trainers.first.train
      resumed.train

      assert_equal trainers.first.state.fetch("coverage"), resumed.state.fetch("coverage")
      first.named_parameters.each { |name, tensor| assert_tensor_close tensor, resumed.model.named_parameters.fetch(name), 1e-5 }
      assert_raises(ArgumentError) { resumed.advance!(mastered: false) }
      assert_raises(ArgumentError) { resumed.advance!(mastered: true) }
    end
  end

  def test_event_mastery_cannot_hide_actor_blind_failures
    rows = generated("test").select { |row| row.dig("world", "profile") == "joint" }
    logits = rows.map do |row|
      world = row.fetch("world")
      actor_case = world.fetch("facts").map { |fact| fact.fetch("actor") }.uniq.size == 2
      prediction = row.fetch("target")
      if actor_case && prediction != "unknown"
        prediction = world.fetch("facts").first.fetch("truth") == world.fetch("assertion") ? "yes" : "no"
      end
      row.fetch("options").map { |option| option.fetch("id") == prediction ? 8.0 : 0.0 }
    end
    value = DecisionJudgment.metrics(rows, logits)
    cell = value.fetch("by_language").fetch("en-US")

    assert_in_delta 0.5, cell.fetch("binding_all_correct")
    assert_in_delta 0.0, cell.fetch("actor_binding_all_correct")
    assert_in_delta 1.0, cell.fetch("event_binding_all_correct")
    refute DecisionJudgment.eligible?(value, "joint", { "mixed_groups" => 0.5, "joint_known_recall" => 0.7, "joint_unknown_recall" => 0.7 })
  end

  def test_accumulated_ce_matches_larger_microbatches_with_three_and_eighteen_options
    Dir.mktmpdir do |directory|
      dataset = write_dataset(File.join(directory, "data.jsonl"), mixture)
      cfg = tiny_config(model: { score_mode: "matching", embedding_norm: true, position_scale: 0.02 }, input: { state_max_tokens: 256, question_option_max_tokens: 256 },
        training: { choice_microbatch: 4, gradient_accumulation: 8, steps: 1, track_coverage: true })
      Torch.manual_seed(1337)
      first = EasyAI::Decision::ChoiceModel.new(cfg)
      second = EasyAI::Decision::ChoiceModel.new(cfg.with(training: { choice_microbatch: 8, gradient_accumulation: 4 }))
      second.load_state_dict(first.state_dict)
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      trainers = [first, second].each_with_index.map { |model, index| EasyAI::Decision::JudgmentTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, index.to_s)) }
      trainers.each { |trainer| trainer.advance!(mastered: true); trainer.train }

      assert_in_delta trainers.first.state.fetch("last_train_loss"), trainers.last.state.fetch("last_train_loss"), 1e-5
      assert_equal trainers.first.state.fetch("coverage"), trainers.last.state.fetch("coverage")
      differences = first.named_parameters.map do |name, tensor|
        other = second.named_parameters.fetch(name)
        [name, (tensor - other).abs.max.item, (tensor.grad - other.grad).abs.max.item]
      end
      # Near-zero, shift-invariant bias gradients can receive different Adam steps
      # after FP32 summation. Compare effective gradients and probability behavior.
      assert differences.all? { |_, _, gradient| gradient < 1e-5 }
      examples = dataset.first(8)
      outputs = [first, second].map { |model| SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: "cpu", batch_size: 4).collect(examples) }
      calibrator = EasyAI::Decision::Calibrator.new
      delta = outputs.first.zip(outputs.last).flat_map do |left, right|
        calibrator.probabilities(left).zip(calibrator.probabilities(right)).map { |a, b| (a - b).abs }
      end.max

      assert_operator delta, :<, 1e-4
    end
  end
end
# rubocop:enable Minitest/MultipleAssertions
