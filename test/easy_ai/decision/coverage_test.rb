require "test_helper"
require_relative "../../../benchmarks/decision/semantic_coverage_evaluation"

class CoverageTest < Minitest::Test
  def test_coverage_counts_repetition_and_shared_groups_separately
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "rows.jsonl"), [example(id: "a"),
        example(id: "b", state: "not red").to_h.merge("group_id" => "a"), example(id: "c", state: "if blue")])
      result = EasyAI::Decision::Data::CoverageAudit.call(data, visits: [2, 1, 0], input_tokens: 99)
      all = result.fetch("strata").fetch("all")

      assert_equal [3, 2, 2, 1, 3], all.values_at("rows", "groups", "seen_rows", "seen_groups", "occurrences")
      assert_in_delta 2.0 / 3, all.fetch("row_coverage")
      assert_equal 3, result.dig("strata", "language/en", "occurrences")
      assert_equal ["all", "source/local", "language/en", "label/local/r"], result.fetch("strata").keys
      assert_equal 99, result.fetch("input_tokens")
    end
  end

  def test_coverage_treats_every_declared_language_the_same_without_text_rules
    Dir.mktmpdir do |dir|
      texts = { "zh" => "不会迟到", "ja" => "遅刻しない", "ko" => "늦지 않아요", "ar" => "لن أتأخر", "und" => "unannotated" }
      rows = texts.map { |language, text| example(id: language, state: text, language: language) }
      dataset = write_dataset(File.join(dir, "rows.jsonl"), rows)
      report = EasyAI::Decision::Data::CoverageAudit.call(dataset, visits: [1, 1, 1, 1, 1])

      texts.each_key do |language|
        assert_equal 1, report.dig("strata", "language/#{language}", "seen_rows")
      end
      assert_equal 1 + 1 + texts.size + 1, report.fetch("strata").size
    end
  end

  def test_token_counts_exclude_padding_and_count_joint_repetition
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    rows = [example, example(id: "long", state: "longer state")]
    %w[separate joint].each do |mode|
      collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: tiny_config(model: { encoding_mode: mode }))
      batch = collator.call(rows)
      actual = mode == "joint" ? batch[:joint_mask].sum.item : batch[:state_mask].sum.item + batch[:option_mask].sum.item

      assert_equal actual, collator.input_token_counts.sum
      assert_operator collator.input_token_counts.last, :>, collator.input_token_counts.first
    end
  end

  def test_coverage_resumes_exactly_and_does_not_change_learning
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "train.jsonl"), [example(id: "a"), example(id: "b", state: "blue", target: "b")])
      cfg = tiny_config(training: { track_coverage: true }, model: { dropout: 0.1 })
      Torch.manual_seed(19)
      full_model = EasyAI::Decision::ChoiceModel.new(cfg)
      other_model = EasyAI::Decision::ChoiceModel.new(cfg.with(training: { track_coverage: false }))
      other_model.load_state_dict(full_model.state_dict)
      factory = ->(model, path) { EasyAI::Decision::Trainer.new(model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: data, output: path) }
      full = factory.call(full_model, File.join(dir, "full"))
      plain = factory.call(other_model, File.join(dir, "plain"))
      full.train(steps: 2)
      checkpoint = full.last_checkpoint
      full.train(steps: 4)
      plain.train(steps: 4)
      resumed = EasyAI::Decision::Trainer.resume(checkpoint, dataset: data, output: File.join(dir, "resume"), device: "cpu")
      resumed.train(steps: 4)

      assert_equal full.state.fetch("coverage"), resumed.state.fetch("coverage")
      assert_equal full.state.fetch("examples_seen"), full.state.dig("coverage", "row_visits").sum
      assert_operator full.state.dig("coverage", "input_tokens"), :>, 0
      full.model.state_dict.each do |name, tensor|
        assert_tensor_close tensor, resumed.model.state_dict.fetch(name), 1e-7
        assert_tensor_close tensor, plain.model.state_dict.fetch(name), 1e-7
      end
    end
  end

  def test_expansion_preserves_facts_labels_and_source_groups
    EasyAI::Decision::Data::SemanticExpansion::OPTIONS.each do |source, options|
      row = EasyAI::Decision::Data::Example.new({ "id" => "row", "group_id" => "family", "source" => source, "language" => "zh-CN",
        "state" => "小林买了票，小周没有买票。", "question" => "根据上下文，以下陈述是否成立：小周买了票。",
        "options" => options.map { |id, text| { "id" => id, "text" => text } }, "target" => "no" })
      expanded = EasyAI::Decision::Data::SemanticExpansion.call(row)

      assert_equal row.to_h.slice("state", "group_id", "language", "source", "target"), expanded.slice("state", "group_id", "language", "source", "target")
      assert_equal row.options.map { |option| option.fetch("id") }, expanded.fetch("options").map { |option| option.fetch("id") }
      assert_includes expanded.fetch("question"), "小周买了票。"
      assert_equal "row", expanded.dig("augmentation", "parent_id")
      refute_equal row.id, expanded.fetch("id")
    end
  end

  def test_expansion_rejects_unrecognized_task_format
    row = example.to_h.merge("source" => "OCNLI", "options" => [
      { "id" => "yes", "text" => "成立" }, { "id" => "no", "text" => "不成立" }, { "id" => "unknown", "text" => "信息不足" }], "target" => "no")

    assert_raises(ArgumentError) { EasyAI::Decision::Data::SemanticExpansion.call(EasyAI::Decision::Data::Example.new(row)) }
    assert_raises(KeyError) { EasyAI::Decision::Data::SemanticExpansion.call(example) }
  end

  def test_batched_audit_matches_public_predictor_and_reordered_candidates
    model = EasyAI::Decision::ChoiceModel.new(tiny_config(model: { dropout: 0.1 }))
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    rows = [example, example(id: "long", state: "longer blue state", options: [
      { id: "r", text: "red" }, { id: "b", text: "blue" }, { id: "g", text: "green" }])]
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer)
    actual = evaluator.collect(rows)
    predictor = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu")
    rows.each_with_index do |row, index|
      predictor.logits(row).each_with_index do |value, option|
        assert_in_delta value, actual[index][option], 1e-5
      end
    end
    report = evaluator.evaluate(rows, permutations: true)

    assert_in_delta 1.0, report.fetch("permutation_agreement")
    assert_equal 2, report.fetch("count")
    assert_equal [2, 3], actual.map(&:size)
  end
end
