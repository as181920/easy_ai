require "test_helper"

class TrainingTest < Minitest::Test
  def test_adamw_matches_libtorch_and_restores_state
    left = Torch::NN::Linear.new(2, 1)
    right = Torch::NN::Linear.new(2, 1)
    right.load_state_dict(left.state_dict)
    reference = Torch::Optim::AdamW.new(left.parameters, lr: 0.001)
    optimizer = EasyAI::Optim::AdamW.new(right.named_parameters, learning_rate: 0.001)
    x = Torch.tensor([[1.0, 2.0]])
    3.times do
      reference.zero_grad
      optimizer.zero_grad
      left.call(x).square.sum.backward
      right.call(x).square.sum.backward
      reference.step
      optimizer.step
    end

    assert_tensor_close left.weight, right.weight
    saved = optimizer.state_dict
    restored = EasyAI::Optim::AdamW.new(right.named_parameters, learning_rate: 0.2).load_state_dict(saved)

    assert_equal optimizer.steps, restored.steps
    assert_in_delta 0.001, restored.learning_rate
  end

  def test_resumed_training_matches_uninterrupted_with_dropout
    Dir.mktmpdir do |dir|
      dataset = write_dataset(File.join(dir, "train.jsonl"), [example(id: "r"), example(id: "b", state: "blue", target: "b")])
      config = tiny_config(model: { dropout: 0.1, embedding_norm: true, position_scale: 0.02 },
        training: { balance_labels: true, resample_negatives: true })
      Torch.manual_seed(55)
      model = EasyAI::Decision::ChoiceModel.new(config)
      other = EasyAI::Decision::ChoiceModel.new(config)
      other.load_state_dict(model.state_dict)
      original = trainer(model, dataset, File.join(dir, "full"))
      original.train(steps: 4)
      partial = trainer(other, dataset, File.join(dir, "resumed"))
      partial.train(steps: 2)
      resumed = EasyAI::Decision::Trainer.resume(partial.last_checkpoint, dataset: dataset, output: File.join(dir, "resumed"), steps: 4, device: "cpu")
      resumed.train

      original.model.state_dict.each do |name, tensor|
        assert_operator (tensor - resumed.model.state_dict.fetch(name)).abs.max.item, :<=, 1e-7, name
      end
      assert_equal 4, resumed.state["step"]
      assert_equal original.optimizer.steps, resumed.optimizer.steps
    end
  end

  def test_small_model_can_learn_a_two_example_task
    Dir.mktmpdir do |dir|
      dataset = write_dataset(File.join(dir, "train.jsonl"), [example(id: "r"), example(id: "b", state: "blue", target: "b")])
      config = tiny_config(model: { embedding_norm: true, position_scale: 0.02, score_mode: "matching" },
        training: { learning_rate: 0.003, choice_microbatch: 4, steps: 100, checkpoint_every: 100 })
      Torch.manual_seed(4)
      instance = trainer(EasyAI::Decision::ChoiceModel.new(config), dataset, dir)
      instance.train
      predictor = EasyAI::Decision::Predictor.new(model: instance.model, tokenizer: instance.tokenizer, device: "cpu")
      results = EasyAI::Decision::Evaluator.new(predictor).evaluate(dataset)

      assert_in_delta(1.0, results["accuracy"])
      assert_operator results["nll"], :<, 0.2
    end
  end

  def test_paired_rotary_training_resumes_exactly
    Dir.mktmpdir do |dir|
      rows = [example(id: "a"), example(id: "b", target: "b")].map do |row|
        row.to_h.merge("group_id" => "family", "contrast_group" => "pair")
      end
      dataset = write_dataset(File.join(dir, "train.jsonl"), rows)
      config = tiny_config(model: { position_encoding: "rotary", dropout: 0.1 }, training: { paired_sampling: true })
      Torch.manual_seed(19)
      full_model = EasyAI::Decision::ChoiceModel.new(config)
      partial_model = EasyAI::Decision::ChoiceModel.new(config)
      partial_model.load_state_dict(full_model.state_dict)
      full = trainer(full_model, dataset, File.join(dir, "full"))
      full.train(steps: 4)
      partial = trainer(partial_model, dataset, File.join(dir, "partial"))
      partial.train(steps: 2)
      resumed = EasyAI::Decision::Trainer.resume(partial.last_checkpoint, dataset: dataset,
        output: File.join(dir, "partial"), steps: 4, device: "cpu")
      resumed.train

      full.model.state_dict.each do |name, tensor|
        assert_tensor_close tensor, resumed.model.state_dict.fetch(name), 1e-7
      end
    end
  end

  def test_recoverable_memory_error_restores_committed_state_on_cpu
    Dir.mktmpdir do |dir|
      dataset = write_dataset(File.join(dir, "train.jsonl"), [example])
      config = tiny_config(training: { choice_microbatch: 1 })
      instance = trainer(EasyAI::Decision::ChoiceModel.new(config), dataset, dir)
      instance.train(steps: 2)
      saved = instance.model.encoder.embedding.weight.detach.clone
      Torch.no_grad { instance.model.encoder.embedding.weight.add!(10) }
      instance.instance_variable_set(:@device, "cuda")
      instance.send(:recover_from_gpu, EasyAI::Runtime::DevicePolicy::MemoryBudgetExceeded.new("injected"))

      assert_equal "cpu", instance.device
      assert_equal 2, instance.state["step"]
      assert_tensor_close saved, instance.model.encoder.embedding.weight
    end
  end

  def test_binding_training_resumes_with_complete_groups_after_microbatch_reduction
    Dir.mktmpdir do |dir|
      rows = 4.times.map do |i|
        example(id: i.to_s, state: i.even? ? "red" : "blue", target: i.even? ? "r" : "b")
          .to_h.merge("group_id" => "family", "contrast_groups" => { "binding" => "four" })
      end
      data = write_dataset(File.join(dir, "train.jsonl"), rows)
      config = tiny_config(model: { dropout: 0.1 }, training: { paired_sampling: true, contrast_strategy: "binding", choice_microbatch: 4 })
      Torch.manual_seed(33)
      instance = trainer(EasyAI::Decision::ChoiceModel.new(config), data, File.join(dir, "full"))
      # Exercise the actual GPU recovery path without requiring CUDA in CI.
      instance.define_singleton_method(:transfer_to) { |_destination| @device = "cpu" }
      error = EasyAI::Runtime::DevicePolicy::MemoryBudgetExceeded.new("simulated capacity failure")
      instance.send(:recover_from_gpu, error)

      assert_equal 2, instance.instance_variable_get(:@microbatch)
      assert_equal 2, instance.instance_variable_get(:@accumulation)
      instance.train(steps: 2)
      resume_path = instance.last_checkpoint
      instance.train(steps: 4)
      resumed = EasyAI::Decision::Trainer.resume(resume_path, dataset: data, output: File.join(dir, "resumed"), steps: 4, device: "cpu")
      resumed.train

      instance.model.state_dict.each { |name, tensor| assert_tensor_close tensor, resumed.model.state_dict.fetch(name), 1e-7 }
      assert_equal 16, resumed.state["examples_seen"]
      assert_equal "binding", resumed.model.config[:training]["contrast_strategy"]
    end
  end

  def test_checkpoint_corruption_is_detected
    Dir.mktmpdir do |dir|
      model = EasyAI::Decision::ChoiceModel.new(tiny_config)
      path = EasyAI::Decision::Checkpoint.save(dir, model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new)
      File.open(File.join(path, "weights.pt"), "ab") { |file| file.write("broken") }
      assert_raises(ArgumentError) { EasyAI::Decision::Checkpoint.load(path) }
    end
  end

  def test_dataset_changes_cannot_silently_resume
    Dir.mktmpdir do |dir|
      first = write_dataset(File.join(dir, "first.jsonl"), [example])
      second = write_dataset(File.join(dir, "second.jsonl"), [example(state: "changed")])
      instance = trainer(EasyAI::Decision::ChoiceModel.new(tiny_config), first, dir)
      assert_raises(ArgumentError) { EasyAI::Decision::Trainer.resume(instance.last_checkpoint, dataset: second, output: dir) }
    end
  end

  def test_automatic_growth_rolls_back_without_repeating_consumed_steps
    Dir.mktmpdir do |dir|
      dataset = write_dataset(File.join(dir, "train.jsonl"), [example(id: "train")])
      validation = write_dataset(File.join(dir, "validation.jsonl"), [example(id: "validation")])
      config = tiny_config(training: { eval_every: 1, steps: 3 },
        growth: { enabled: true, patience: 1, min_delta: 100.0, trial_evaluations: 1, max_trials: 1 })
      instance = EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, validation: validation, output: dir)
      instance.train

      expected_examples = 3 * config[:training]["choice_microbatch"] * config[:training]["gradient_accumulation"]

      assert_equal [3, expected_examples], instance.state.values_at("step", "examples_seen")
      assert_equal config[:model]["encoder_layers"], instance.model.config[:model]["encoder_layers"]
      assert_equal %w[trial_started rejected], instance.state["growth"]["events"].map { |event| event["action"] }
      assert_nil instance.state["growth"]["pending"]
      resumed = EasyAI::Decision::Trainer.resume(instance.last_checkpoint, dataset: dataset,
        validation: validation, output: dir, steps: 4, device: "cpu")
      resumed.train

      assert_equal [4, 1], [resumed.state["step"], resumed.state["growth"]["trials"]]
    end
  end

  def test_validation_selects_best_weights_and_stops_without_losing_resume_state
    Dir.mktmpdir do |dir|
      dataset = write_dataset(File.join(dir, "train.jsonl"), [example(id: "train")])
      validation = write_dataset(File.join(dir, "validation.jsonl"), [example(id: "validation")])
      config = tiny_config(training: { steps: 10, eval_every: 1, early_stopping_patience: 2 })
      instance = EasyAI::Decision::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config),
        tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, validation: validation, output: dir)
      losses = [0.7, 0.5, 0.6, 0.8]
      instance.define_singleton_method(:validation_loss) { losses.shift }
      instance.train
      best = EasyAI::Decision::Checkpoint.load(instance.state.fetch("best_checkpoint"))
      latest = EasyAI::Decision::Checkpoint.load(instance.last_checkpoint)

      assert_equal [4, 2], instance.state.values_at("step", "best_step")
      assert_equal 2, best[:metadata]["training"]["step"]
      assert_nil best[:metadata]["optimizer"]
      refute_nil latest[:metadata]["optimizer"]
      resumed = EasyAI::Decision::Trainer.resume(instance.last_checkpoint, dataset: dataset,
        validation: validation, output: dir, steps: 5, device: "cpu")
      resumed.train

      assert_equal 5, resumed.state["step"]
    end
  end

  private

  def trainer(model, dataset, output)
    EasyAI::Decision::Trainer.new(model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: output)
  end
end
