require_relative "../../test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/relational_encoding"

class RelationalEncodingTest < Minitest::Test
  def parent(config = tiny_config)
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    { model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: tokenizer, weights_fingerprint: "fixture" }
  end

  def test_evaluation_collection_schedule_preserves_logits_and_live_parameters
    rows = Array.new(13) { |index| example(id: index.to_s, state: "red " * (index + 1)) }
    %w[separate joint].each do |mode|
      source = parent(tiny_config(model: { encoding_mode: mode, position_encoding: "rotary" }))
      model = source.fetch(:model)
      identities = model.named_parameters.transform_values(&:object_id)
      original = model.state_dict.transform_values { |tensor| tensor.detach.clone }
      first = SemanticCoverageEvaluation.new(model: model, tokenizer: source.fetch(:tokenizer), batch_size: 4, gc_interval: 1).collect(rows)
      other = SemanticCoverageEvaluation.new(model: model, tokenizer: source.fetch(:tokenizer), batch_size: 4, gc_interval: 8).collect(rows)

      assert_equal first, other
      assert_equal identities, model.named_parameters.transform_values(&:object_id)
      assert model.state_dict.all? { |key, tensor| (tensor - original.fetch(key)).abs.max.item.zero? }
    end
  end

  def test_evaluation_rejects_invalid_collection_intervals
    source = parent

    [0, -1, 1.5, "8"].each do |interval|
      assert_raises(ArgumentError) { SemanticCoverageEvaluation.new(model: source.fetch(:model), tokenizer: source.fetch(:tokenizer), gc_interval: interval) }
    end
  end

  def test_joint_transfer_copies_all_active_tensors_and_declares_unused_modules
    source = parent(tiny_config(model: { score_mode: "matching" }))
    config = source.fetch(:model).config.with(model: { encoding_mode: "joint" })
    model, manifest = EasyAI::Decision::WeightTransfer.apply(parent: source, config: config, tokenizer: source.fetch(:tokenizer), seed: 1)
    expected = source.fetch(:model).state_dict

    assert_empty manifest.fetch("new")
    assert manifest.fetch("unused").any? { |key| key.start_with?("interactions.") }
    assert manifest.fetch("unused").any? { |key| key.start_with?("matching.") }
    assert model.state_dict.all? { |key, value| (expected.fetch(key) - value).abs.max.item.zero? }
    assert_equal model.parameter_count, manifest.fetch("parameters")
  end

  def test_transfer_rejects_vocabulary_and_unknown_architecture_changes
    source = parent
    config = source.fetch(:model).config
    %i[vocab_size hidden_size encoder_layers].each do |key|
      value = config[:model].fetch(key.to_s) * 2
      assert_raises(ArgumentError) do
        EasyAI::Decision::WeightTransfer.apply(parent: source, config: config.with(model: { key => value }), tokenizer: source.fetch(:tokenizer), seed: 1)
      end
    end
    assert_raises(ArgumentError) do
      EasyAI::Decision::WeightTransfer.apply(parent: source, config: config.with(model: { evidence_head: true }), tokenizer: source.fetch(:tokenizer), seed: 1)
    end
  end

  def test_fit_gate_cannot_hide_actor_failure_behind_event_success
    rules = { "balanced" => 0.99, "mixed_groups" => 0.9 }
    cell = { "balanced" => 1.0, "actor_binding_groups" => 2, "event_binding_groups" => 2,
      "actor_binding_all_correct" => 1.0, "event_binding_all_correct" => 1.0 }
    value = { "by_language" => { "en-US" => cell.dup, "zh-CN" => cell.dup } }

    assert DecisionRelational.mastered?(value, rules)
    value.fetch("by_language").fetch("zh-CN")["actor_binding_all_correct"] = 0.0

    refute DecisionRelational.mastered?(value, rules)
    value.fetch("by_language").delete("zh-CN")

    refute DecisionRelational.mastered?(value, rules)
  end

  def test_joint_answer_spans_and_padding_are_aligned_for_variable_candidate_counts
    source = parent(tiny_config(model: { encoding_mode: "joint", pooling: "candidate" }))
    model, tokenizer = source.values_at(:model, :tokenizer)
    first = example(state: "red blue", options: [{ id: 1, text: "red" }, { id: "b", text: "blue" }], target: 1)
    second = example(id: "long", state: "a longer record with more padding", options: [{ id: "r", text: "red" }, { id: 2, text: "blue" }, { id: "g", text: "green" }])
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: model.config)
    batch = collator.call([first, second])
    selected = batch.fetch(:joint_ids)[0][0].to_a.zip(batch.fetch(:joint_answer_mask)[0][0].to_a).filter_map { |id, selected| id if selected }

    assert_equal tokenizer.encode("red"), selected
    assert_equal [true, true, false], batch.fetch(:candidate_mask)[0].to_a
    model.eval
    alone, mixed = Torch.no_grad { [model.call(collator.call([first]))[0].to_a, model.call(batch)[0].to_a.first(2)] }

    assert_operator alone.zip(mixed).map { |a, b| (a - b).abs }.max, :<, 1e-5
  end
  def test_positions_break_mean_pooling_permutation_symmetry
    Torch.manual_seed(1337)
    encoder = EasyAI::Decision::Encoder.new(tiny_config)
    encoder.eval
    ids = Torch.tensor([[6, 7, 8], [8, 7, 6]], dtype: :int64)
    mask = Torch.ones([2, 3], dtype: :bool)
    with_positions = Torch.no_grad { encoder.call(ids, mask: mask).mean(dim: 1) }
    encoder.instance_variable_set(:@position_scale, 0.0)
    without_positions = Torch.no_grad { encoder.call(ids, mask: mask).mean(dim: 1) }

    assert_operator (with_positions[0] - with_positions[1]).abs.max.item, :>, 1e-5
    assert_tensor_close without_positions[0], without_positions[1], 1e-5
  end

  def test_both_encodings_have_effective_gradients_and_exact_checkpoint_resume
    %w[separate joint].each do |mode|
      Dir.mktmpdir do |directory|
        dataset = write_dataset(File.join(directory, "data.jsonl"), [example(id: "a"), example(id: "b", state: "blue", target: "b")])
        source = parent(tiny_config(model: { encoding_mode: mode }, training: { steps: 2 }))
        trainer = EasyAI::Decision::CandidateTrainer.new(model: source.fetch(:model), tokenizer: source.fetch(:tokenizer), dataset: dataset, output: File.join(directory, "first"))
        before = trainer.model.encoder.embedding.weight.detach.clone
        trainer.train(steps: 1)
        resumed = EasyAI::Decision::CandidateTrainer.resume(trainer.last_checkpoint, dataset: dataset, output: File.join(directory, "resumed"))

        assert_operator (before - trainer.model.encoder.embedding.weight).abs.max.item, :>, 0.0
        assert_operator trainer.model.encoder.embedding.weight.grad.abs.max.item, :>, 0.0
        trainer.train
        resumed.train

        assert trainer.model.state_dict.all? { |key, value| (value - resumed.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      end
    end
  end
  def test_joint_inference_preserves_ids_probabilities_and_candidate_permutation_across_chunks
    source = parent(tiny_config(model: { encoding_mode: "joint", pooling: "candidate" }))
    options = [{ id: 1, text: "red" }, { id: "blue", text: "blue" }, { id: 3, text: "green" }]
    predictor = EasyAI::Decision::Predictor.new(model: source.fetch(:model), tokenizer: source.fetch(:tokenizer), device: :cpu, candidate_chunk_size: 1)
    first = predictor.probabilities(state: "red blue", question: "Color?", options: options)
    other = EasyAI::Decision::Predictor.new(model: source.fetch(:model), tokenizer: source.fetch(:tokenizer), device: "cpu", candidate_chunk_size: 3)
    second = other.probabilities(state: "red blue", question: "Color?", options: options.reverse)
    probabilities = JSON.parse(JSON.generate(first)).fetch("probabilities")

    assert_equal %w[1 blue 3], probabilities.keys
    assert_in_delta 1.0, probabilities.values.sum, 1e-6
    assert probabilities.all? { |key, value| (value - second.fetch("probabilities").fetch(key)).abs < 1e-5 }
  end

  def test_joint_accumulation_matches_effective_batch_gradients
    Dir.mktmpdir do |directory|
      dataset = write_dataset(File.join(directory, "data.jsonl"), [example(id: "a"), example(id: "b", state: "blue", target: "b")])
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      Torch.manual_seed(1337)
      config = tiny_config(model: { encoding_mode: "joint" }, training: { steps: 1, choice_microbatch: 2, gradient_accumulation: 2 })
      first = EasyAI::Decision::ChoiceModel.new(config)
      second = EasyAI::Decision::ChoiceModel.new(config.with(training: { choice_microbatch: 4, gradient_accumulation: 1 }))
      second.load_state_dict(first.state_dict)
      trainers = [first, second].each_with_index.map do |model, index|
        EasyAI::Decision::CandidateTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, index.to_s))
      end
      trainers.each(&:train)
      differences = first.named_parameters.map { |key, tensor| (tensor.grad - second.named_parameters.fetch(key).grad).abs.max.item }

      assert_operator differences.max, :<, 1e-5
      assert_in_delta trainers.first.state.fetch("last_train_loss"), trainers.last.state.fetch("last_train_loss"), 1e-5
    end
  end
  def test_followups_and_confirmation_require_measured_prior_results
    Dir.mktmpdir do |root|
      DecisionRelational.stubs(:verify).returns({})
      assert_raises(RuntimeError) { DecisionRelational.freeze_recipe(root, "joint-candidate") }
      assert_raises(ArgumentError) { DecisionRelational.freeze_recipe(root, "unbounded-sweep") }
      DecisionRelational.write(File.join(root, "fit-selection.json"), { "winner" => "joint", "results_sha256" => {} })
      assert_raises(RuntimeError) { DecisionRelational.check_confirmation!(root, "separate") }
      DecisionRelational.check_confirmation!(root, "joint")
    end
  end
  def test_transfer_rejects_changed_tokenizer_and_does_not_alias_parent_weights
    source = parent
    tokenizer = source.fetch(:tokenizer).dup
    tokenizer.stubs(:fingerprint).returns("different-vocabulary")

    assert_raises(ArgumentError) do
      EasyAI::Decision::WeightTransfer.apply(parent: source, config: source.fetch(:model).config, tokenizer: tokenizer, seed: 1)
    end
    model, = EasyAI::Decision::WeightTransfer.apply(parent: source, config: source.fetch(:model).config, tokenizer: source.fetch(:tokenizer), seed: 1)
    before = source.fetch(:model).encoder.embedding.weight.detach.clone
    Torch.no_grad { model.encoder.embedding.weight.add!(1.0) }

    assert_tensor_close before, source.fetch(:model).encoder.embedding.weight
  end

  def test_failed_fits_select_no_checkpoint_and_keep_acceptance_closed
    Dir.mktmpdir do |root|
      DecisionRelational.stubs(:verify).returns({})
      %w[separate joint].each do |arm|
        FileUtils.mkdir_p(File.join(root, "fit-#{arm}-1337"))
        DecisionRelational.write(File.join(root, "fit-#{arm}-1337/result.json"), { "passed" => false })
      end
      result = DecisionRelational.select_fit(root)

      assert_nil result.fetch("winner")
      assert_nil result.fetch("checkpoint")
      refute_path_exists File.join(root, "acceptance-opened.json")
      assert_raises(RuntimeError) { DecisionRelational.select_fit(root) }
    end
  end
  def test_encoder_token_accounting_matches_hand_calculated_padding_and_state_repetition
    lengths = [{ "state" => 4, "options" => [3, 5] }, { "state" => 7, "options" => [6, 2] }]

    assert_equal({ "valid" => 27, "padded" => 38 }, DecisionRelational.batch_token_counts(lengths, :separate))
    assert_equal({ "valid" => 34, "padded" => 48 }, DecisionRelational.batch_token_counts(lengths, "joint"))
    assert_raises(ArgumentError) { DecisionRelational.batch_token_counts(lengths, "invalid") }
  end
end
