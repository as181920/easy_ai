require "test_helper"

class BindingTest < Minitest::Test
  def test_subject_switch_changes_answer_only_for_mixed_facts
    rows = relation_rows
    rows.group_by { |row| row.dig("relation", "checks", "subject_switch") }.each_value do |pair|
      assert_equal 2, pair.size
      left, right = pair

      assert_equal left["state"], right["state"]
      refute_equal left.dig("relation", "subject"), right.dig("relation", "subject")
      assert_equal left.dig("relation", "facts").uniq.size == 2, left["target"] != right["target"]
    end
  end

  def test_role_swap_reverses_facts_and_answer_for_the_same_question
    mixed = relation_rows.select { |row| row.dig("relation", "facts").uniq.size == 2 }
    mixed.group_by { |row| row.dig("relation", "checks", "role_swap") }.each_value do |pair|
      assert_equal 2, pair.size
      left, right = pair

      assert_equal left["question"], right["question"]
      assert_equal left.dig("relation", "facts"), right.dig("relation", "facts").reverse
      refute_equal left["target"], right["target"]
    end
  end

  def test_binding_groups_balance_subjects_and_labels_within_each_pattern
    groups = relation_rows.group_by { |row| row.dig("contrast_groups", "binding") }.values

    assert_equal 16, groups.size
    groups.each do |group|
      assert_equal 4, group.size
      assert_equal [0, 1], group.map { |row| row.dig("relation", "subject") }.uniq.sort
      assert_equal [2, 2], group.map { |row| row["target"] }.tally.values.sort
      assert_equal 1, group.map { |row| row.dig("relation", "facts").uniq.size }.uniq.size
    end
  end

  def test_group_metadata_round_trips_without_becoming_input_text
    row = EasyAI::Decision::Data::Example.new(relation_rows.first)
    restored = EasyAI::Decision::Data::Example.new(row.to_h)

    assert_equal row.contrast_groups, restored.contrast_groups
    assert_equal [row.state, row.question, *row.options.map { |option| option["text"] }], restored.texts
    assert_raises(ArgumentError) { EasyAI::Decision::Data::Example.new(row.to_h.merge("contrast_groups" => [])) }
    assert_raises(ArgumentError) { EasyAI::Decision::Data::Example.new(row.to_h.merge("contrast_groups" => { "binding" => 7 })) }
  end

  def test_sampler_keeps_complete_reproducible_groups
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "train.jsonl"), relation_rows)
      sampler = EasyAI::Decision::Data::PairSampler.new(data, strategy: "binding")
      indices = sampler.sample(32, rng: Random.new(1337))

      assert_equal indices, sampler.sample(32, rng: Random.new(1337))
      indices.each_slice(4) do |group|
        assert_equal 4, group.uniq.size
        assert_equal 1, group.map { |i| data[i].contrast_groups.fetch("binding") }.uniq.size
        assert_equal 1, group.map { |i| data[i].language }.uniq.size
        assert_equal [2, 2], group.map { |i| data[i].target }.tally.values.sort
      end
    end
  end

  def test_sampler_rejects_partial_groups_and_partial_update_sizes
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "train.jsonl"), relation_rows)
      sampler = EasyAI::Decision::Data::PairSampler.new(data, strategy: "binding")

      assert_raises(ArgumentError) { sampler.sample(6, rng: Random.new(1)) }
      incomplete = write_dataset(File.join(dir, "incomplete.jsonl"), relation_rows.drop(1))

      assert_raises(ArgumentError) { EasyAI::Decision::Data::PairSampler.new(incomplete, strategy: "binding") }
    end
  end

  def test_config_rejects_invalid_contrast_settings
    assert_raises(ArgumentError) { tiny_config(training: { paired_sampling: true, contrast_strategy: "binding", choice_microbatch: 2 }) }
    assert_raises(ArgumentError) { tiny_config(training: { contrast_strategy: "binding", choice_microbatch: 4 }) }
    assert_raises(ArgumentError) { tiny_config(training: { contrast_strategy: "unknown" }) }
  end

  def test_metrics_expose_constant_answer_shortcut
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "train.jsonl"), relation_rows)
      predictor = EasyAI::Decision::Predictor.new(model: EasyAI::Decision::ChoiceModel.new(tiny_config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, device: "cpu")
      evaluator = EasyAI::Decision::RelationEvaluation.new(predictor)
      evaluator.define_singleton_method(:collect) do |examples|
        examples.map { |row| row.options.map { |option| option["id"] == "yes" ? 5.0 : -5.0 } }
      end
      result = evaluator.evaluate(data)
      switch = result["pairs"]["subject_switch"]

      assert_in_delta 1.0, switch["same_truth"]["expected_relation_rate"]
      assert_in_delta 0.0, switch["mixed_truth"]["expected_relation_rate"]
      assert_in_delta 0.0, result["pairs"]["role_swap"]["both_correct"]
      assert_in_delta 0.0, result["groups"]["binding"]["all_correct"]
      assert result["by_language"].values.all? { |part| part.dig("groups", "binding", "all_correct").zero? }
    end
  end

  def test_v2_metadata_fails_before_model_inference_with_migration_message
    Dir.mktmpdir do |dir|
      predictor = EasyAI::Decision::Predictor.new(model: EasyAI::Decision::ChoiceModel.new(tiny_config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, device: "cpu")
      evaluator = EasyAI::Decision::RelationEvaluation.new(predictor)
      evaluator.define_singleton_method(:collect) { |_examples| raise "Inference should not run" }
      legacy = relation_rows.map do |row|
        row.merge("relation" => row["relation"].except("binding_group"))
      end
      old_data = write_dataset(File.join(dir, "legacy.jsonl"), legacy)
      error = assert_raises(ArgumentError) { evaluator.evaluate(old_data) }

      assert_match(/v3 metadata/, error.message)
    end
  end

  def test_representative_warmup_covers_actions_and_people_without_split_leakage
    Dir.mktmpdir do |dir|
      corpus = EasyAI::Decision::Data::RelationCorpus.new
      output = File.join(dir, "coverage")
      manifest = corpus.write(output: output, sanity_families: 8)
      warmup = EasyAI::Decision::Data::Dataset.new(File.join(output, "sanity.jsonl"))
      train = EasyAI::Decision::Data::Dataset.new(File.join(output, "train.jsonl"))
      held_out = %w[validation calibration test].map { |split| EasyAI::Decision::Data::Dataset.new(File.join(output, "#{split}.jsonl")) }
      EasyAI::Decision::Data::Dataset.assert_disjoint!(warmup, *held_out)
      families = manifest.fetch("sanity_families")

      assert_equal [512, 8], [warmup.size, warmup.groups.size]
      assert_equal (0...8).to_a, families.flat_map { |family| family.first(2) }.uniq.sort
      assert_equal (0...6).to_a, families.map(&:last).uniq.sort
      assert_empty warmup.map(&:id) - train.map(&:id)
    end
  end

  def test_representative_warmup_rejects_invalid_family_counts
    Dir.mktmpdir do |dir|
      corpus = EasyAI::Decision::Data::RelationCorpus.new

      assert_raises(ArgumentError) { corpus.write(output: File.join(dir, "zero"), sanity_families: 0) }
      assert_raises(ArgumentError) { corpus.write(output: File.join(dir, "too_many"), sanity_families: 109) }
    end
  end

  private

  def relation_rows
    EasyAI::Decision::Data::RelationCorpus.new.examples([0, 1, 0], style: 0)
  end
end
