require "test_helper"

class RelationTest < Minitest::Test
  def test_boolean_world_has_complete_balanced_contrast_pairs
    rows = EasyAI::Decision::Data::RelationCorpus.new.examples([0, 1, 0], style: 0)

    assert_equal 64, rows.size
    assert_equal({ "yes" => 32, "no" => 32 }, rows.map { |row| row["target"] }.tally)
    rows.each do |row|
      logic = row.fetch("relation")

      assert_equal logic["facts"][logic["subject"]] == logic["assertion"] ? "yes" : "no", row["target"]
    end
    %w[question_flip fact_flip irrelevant_fact order].each do |kind|
      pairs = rows.group_by { |row| row["relation"]["checks"][kind] }.values

      assert pairs.all? { |pair| pair.size == 2 }
      assert pairs.all? { |a, b| (a["target"] != b["target"]) == %w[question_flip fact_flip].include?(kind) }
    end
  end

  def test_candidate_pooling_excludes_question_special_tokens_and_padding
    config = tiny_config(model: { pooling: "candidate" })
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    rows = [example, example(id: "three", options: [{ id: "r", text: "red" }, { id: "b", text: "blue" }, { id: "g", text: "green" }])]
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config).call(rows)
    selected = batch[:answer_mask].to_a
    ids = batch[:option_ids].to_a

    rows.each_with_index do |row, i|
      row.options.each_with_index do |option, j|
        content = ids[i][j].each_index.filter_map { |k| ids[i][j][k] if selected[i][j][k] }

        assert_equal option["text"], tokenizer.decode(content)
      end
    end

    refute_predicate selected[0][2], :any?
    model = EasyAI::Decision::ChoiceModel.new(config)
    Torch::NN::Functional.cross_entropy(model.call(batch), batch[:targets]).backward

    assert model.parameters.all? { |parameter| parameter.grad.nil? || parameter.grad.abs.max.item.finite? }
  end

  def test_candidate_pooling_keeps_chunk_permutation_and_question_gradients
    config = tiny_config(model: { pooling: "candidate", embedding_norm: true, position_scale: 0.02, score_mode: "matching" })
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    model = EasyAI::Decision::ChoiceModel.new(config)
    row = example
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config).call([row])
    Torch::NN::Functional.cross_entropy(model.call(batch), batch[:targets]).backward
    # Capital C appears only in the question, not in state or candidates.
    gradient = model.encoder.embedding.weight.grad[tokenizer.encode("C").first].abs.sum.item
    full = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu", candidate_chunk_size: 16)
    chunked = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu", candidate_chunk_size: 1)

    assert_operator gradient, :>, 0
    full.logits(row).zip(chunked.logits(row)).each { |a, b| assert_in_delta a, b, 1e-6 }
  end

  def test_family_splits_and_template_holdout_do_not_overlap_training
    Dir.mktmpdir do |dir|
      output = File.join(dir, "relations")
      EasyAI::Decision::Data::RelationCorpus.new.write(output: output, vocab_size: 400)
      datasets = %w[train validation calibration test].map { |split| EasyAI::Decision::Data::Dataset.new(File.join(output, "#{split}.jsonl")) }
      EasyAI::Decision::Data::Dataset.assert_disjoint!(*datasets)
      styles = %w[train test].map do |split|
        File.foreach(File.join(output, "#{split}.jsonl")).map { |line| JSON.parse(line)["relation"]["style"] }.uniq.sort
      end

      assert_equal [[0, 1], [3]], styles
      assert_equal 108, datasets.first.groups.size
      assert_equal 20, datasets.last.groups.size
    end
  end

  def test_joint_encoding_matches_chunked_inference_without_state_cache
    config = tiny_config(model: { encoding_mode: "joint", pooling: "candidate" })
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    model = EasyAI::Decision::ChoiceModel.new(config)
    row = example
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config).call([row])
    Torch::NN::Functional.cross_entropy(model.call(batch), batch[:targets]).backward
    full = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu")
    chunked = EasyAI::Decision::Predictor.new(model: model, tokenizer: tokenizer, device: "cpu", candidate_chunk_size: 1)

    full.logits(row).zip(chunked.logits(row)).each { |a, b| assert_in_delta a, b, 1e-6 }
    full.logits(row)

    assert_equal 0, full.cache_hits
    assert_operator model.encoder.embedding.weight.grad.abs.sum.item, :>, 0
  end

  def test_pair_sampler_keeps_complementary_examples_together
    Dir.mktmpdir do |dir|
      rows = EasyAI::Decision::Data::RelationCorpus.new.examples([0, 1, 0], style: 0)
      dataset = write_dataset(File.join(dir, "pairs.jsonl"), rows)
      sampler = EasyAI::Decision::Data::PairSampler.new(dataset)
      indices = sampler.sample(32, rng: Random.new(5))

      assert_equal indices, sampler.sample(32, rng: Random.new(5))
      indices.each_slice(2) do |a, b|
        assert_equal dataset[a].contrast_group, dataset[b].contrast_group
        refute_equal dataset[a].target, dataset[b].target
      end
      assert_raises(ArgumentError) { sampler.sample(3, rng: Random.new(5)) }
      incomplete = write_dataset(File.join(dir, "incomplete.jsonl"), rows.drop(1))

      assert_raises(ArgumentError) { EasyAI::Decision::Data::PairSampler.new(incomplete) }
    end
  end

  def test_pair_metrics_distinguish_accuracy_from_relation_consistency
    Dir.mktmpdir do |dir|
      rows = EasyAI::Decision::Data::RelationCorpus.new.examples([0, 1, 0], style: 0)
      dataset = write_dataset(File.join(dir, "pairs.jsonl"), rows)
      model = EasyAI::Decision::ChoiceModel.new(tiny_config)
      predictor = EasyAI::Decision::Predictor.new(model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new, device: "cpu")
      evaluator = EasyAI::Decision::RelationEvaluation.new(predictor)
      evaluator.define_singleton_method(:collect) do |examples|
        examples.map { |row| row.options.map { |option| option["id"] == row.target ? 5.0 : -5.0 } }
      end
      oracle = evaluator.evaluate(dataset)

      assert_equal [1.0, 0.0], oracle.values_at("accuracy", "maximum_permutation_logit_error")
      assert oracle["pairs"].values.all? { |pair| pair["both_correct"] == 1.0 && pair["expected_relation_rate"] == 1.0 }
      evaluator.define_singleton_method(:collect) do |examples|
        examples.map { |row| row.options.map { |option| option["id"] == "yes" ? 5.0 : -5.0 } }
      end
      constant = evaluator.evaluate(dataset)

      assert_equal [0.5, 0.0, 0.0], [constant["accuracy"], *constant["pairs"]["question_flip"].values_at("both_correct", "expected_relation_rate")]
      assert_in_delta 1.0, constant["pairs"]["irrelevant_fact"]["expected_relation_rate"]
      incomplete = write_dataset(File.join(dir, "incomplete.jsonl"), rows.drop(1))

      assert_raises(ArgumentError) { evaluator.evaluate(incomplete) }
    end
  end

  def test_bag_of_facts_shortcut_is_visible_in_pattern_metrics
    Dir.mktmpdir do |dir|
      rows = EasyAI::Decision::Data::RelationCorpus.new.examples([0, 1, 0], style: 0)
      rows.each { |row| row["options"].sort_by! { |option| option["id"] == "yes" ? 0 : 1 } }
      same_truth = rows.to_h { |row| [row["id"], row["relation"]["facts"].uniq.length == 1] }
      data = write_dataset(File.join(dir, "pairs.jsonl"), rows)
      predictor = EasyAI::Decision::Predictor.new(model: EasyAI::Decision::ChoiceModel.new(tiny_config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, device: "cpu")
      evaluator = EasyAI::Decision::RelationEvaluation.new(predictor)
      evaluator.define_singleton_method(:collect) do |examples|
        examples.map do |row|
          same_truth[row.id] ? row.options.map { |option| option["id"] == row.target ? 10.0 : -10.0 } : [0.0, 0.0]
        end
      end
      result = evaluator.evaluate(data)

      assert_in_delta 0.75, result["accuracy"]
      assert_in_delta Math.log(2) / 2, result["nll"], 1e-8
      assert_equal [32, 1.0], result["by_fact_pattern"]["same_truth"].values_at("count", "accuracy")
      assert_equal [32, 0.5], result["by_fact_pattern"]["mixed_truth"].values_at("count", "accuracy")
    end
  end
end
