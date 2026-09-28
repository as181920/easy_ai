require "test_helper"

class ModelTest < Minitest::Test
  def setup
    Torch.manual_seed(3)
    @config = tiny_config
    @tokenizer = EasyAI::Tokenizers::ByteBpe.new
    @model = EasyAI::Decision::ChoiceModel.new(@config).eval
    @collator = EasyAI::Decision::Data::Collator.new(tokenizer: @tokenizer, config: @config)
  end

  def test_padding_and_other_examples_do_not_change_a_score
    alone = @model.call(@collator.call([example]))
    together = @model.call(@collator.call([example, example(id: "long", state: "a much longer color description")]))

    assert_tensor_close alone[0], together[0], 2e-6
  end

  def test_permutation_and_chunking_preserve_probabilities
    options = [{ id: "r", text: "red" }, { id: "b", text: "blue" }, { id: "g", text: "green" }]
    full = EasyAI::Decision::Predictor.new(model: @model, tokenizer: @tokenizer, device: "cpu", candidate_chunk_size: 16)
    chunked = EasyAI::Decision::Predictor.new(model: @model, tokenizer: @tokenizer, device: "cpu", candidate_chunk_size: 1)
    first = full.probabilities(state: "red", question: "Color?", options: options)["probabilities"]
    second = chunked.probabilities(state: "red", question: "Color?", options: options.reverse)["probabilities"]

    options.each { |option| assert_in_delta first[option[:id]], second[option[:id]], 1e-6 }
    assert_in_delta 1.0, first.values.sum, 1e-12
  end

  def test_state_cache_reuse_across_questions
    predictor = EasyAI::Decision::Predictor.new(model: @model, tokenizer: @tokenizer, device: "cpu")
    options = [{ id: "a", text: "yes" }, { id: "b", text: "no" }]
    predictor.probabilities(state: "red", question: "Red?", options: options)
    predictor.probabilities(state: "red", question: "Blue?", options: options)

    assert_equal 1, predictor.cache_hits
  end

  def test_variable_candidate_counts_have_finite_gradients
    longer = example(id: "three", options: [{ id: "r", text: "red" }, { id: "b", text: "blue" }, { id: "g", text: "green" }])
    batch = @collator.call([example, longer])
    logits = @model.call(batch)
    loss = Torch::NN::Functional.cross_entropy(logits, batch[:targets])
    loss.backward

    assert_predicate loss.item, :finite?
    assert_equal(-Float::INFINITY, logits[0][2].item)
    @model.parameters.each { |parameter| assert_predicate parameter.grad.abs.max.item, :finite? if parameter.grad }
  end

  def test_masked_language_model_uses_selected_positions
    masker = EasyAI::Decision::Data::Masking.new(tokenizer: @tokenizer, config: @config)
    batch = masker.call([{ "text" => "你好世界" }, { "text" => "hello" }], seed: 4)
    logits = @model.mlm_logits(batch[:ids], mask: batch[:mask], positions: batch[:positions])
    loss = Torch::NN::Functional.cross_entropy(logits, batch[:targets])
    loss.backward

    assert_equal batch[:targets].shape[0], logits.shape[0]
    assert_equal 262, logits.shape[1]
    assert_operator @model.encoder.embedding.weight.grad.abs.sum.item, :>, 0
    assert batch[:targets].to_a.all? { |id| id >= 6 }
  end

  def test_truncation_is_explicit
    assert_raises(ArgumentError) { @collator.call([example(state: "a" * 100)]) }
  end

  def test_invalid_candidate_contract
    assert_raises(ArgumentError) { example(options: [{ id: "r", text: "red" }, { id: "r", text: "blue" }]) }
    assert_raises(ArgumentError) { example(target: "missing") }
  end

  def test_numeric_ids_are_normalized_for_predictions_and_training
    options = [{ id: 0, text: "red" }, { id: "blue", text: "blue" }, { id: 2.5, text: "green" }]
    row = example(options: options, target: 0)
    predictor = EasyAI::Decision::Predictor.new(model: @model, tokenizer: @tokenizer, device: "cpu")
    result = predictor.probabilities(state: "red", question: "Color?", options: options)

    assert_equal 0, row.target_index
    assert_equal %w[0 blue 2.5], result.fetch("probabilities").keys
    assert_equal result, JSON.parse(JSON.generate(result))
    assert_equal 0, options.first[:id]
  end

  def test_matching_scores_preserve_candidate_chunking_and_state_gradients
    model = EasyAI::Decision::ChoiceModel.new(@config.with(model: { embedding_norm: true, position_scale: 0.02, score_mode: "matching" })).eval
    full = EasyAI::Decision::Predictor.new(model: model, tokenizer: @tokenizer, device: "cpu", candidate_chunk_size: 8)
    chunked = EasyAI::Decision::Predictor.new(model: model, tokenizer: @tokenizer, device: "cpu", candidate_chunk_size: 1)
    row = example
    batch = @collator.call([row])
    memory = model.encode_state(batch[:state_ids], batch[:state_mask]).detach.requires_grad!(true)
    logits = model.score_candidates(memory, batch[:state_mask], batch[:option_ids], batch[:option_mask], batch[:candidate_mask])
    Torch::NN::Functional.cross_entropy(logits, batch[:targets]).backward

    full.logits(row).zip(chunked.logits(row)).each { |left, right| assert_in_delta left, right, 1e-6 }
    assert_operator memory.grad.abs.sum.item, :>, 0
  end

  def test_normalized_id_collisions_and_non_json_numbers_are_rejected
    assert_raises(ArgumentError) { example(options: [{ id: 1, text: "red" }, { id: "1", text: "blue" }], target: 1) }
    [true, nil, Float::NAN, Float::INFINITY].each do |id|
      assert_raises(ArgumentError) { example(options: [{ id: id, text: "red" }, { id: "b", text: "blue" }]) }
    end
  end
end
