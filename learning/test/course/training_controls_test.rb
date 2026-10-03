require_relative "../test_helper"

class TrainingControlsTest < Minitest::Test
  def test_weighted_microbatches_match_full_batch_gradient
    p = Torch::NN::Parameter.new(Torch.tensor([1.0]))
    x = Torch.tensor([1.0, 2.0, 3.0])
    (p * x).square.mean.backward
    reference = p.grad.item
    p.grad.zero!
    objectives = [-> { (p * x.narrow(0, 0, 1)).square.mean }, -> { (p * x.narrow(0, 1, 2)).square.mean }]
    EasyAILearning::Training::Accumulation.backward(objectives, counts: [1, 2])

    assert_in_delta reference, p.grad.item, 1e-6
  end

  def test_early_stopping_restores_best_snapshot
    model = EasyAILearning::BasicNN::Mlp.new
    stopping = EasyAILearning::Training::EarlyStopping.new(patience: 2)
    expected = model.hidden.weight.clone

    refute stopping.observe(1.0, model)
    Torch.no_grad { model.hidden.weight.zero! }

    refute stopping.observe(1.1, model)
    assert stopping.observe(1.2, model)
    stopping.restore(model)

    assert_equal expected.to_a, model.hidden.weight.to_a
  end

  def test_loss_scaling_matches_unscaled_gradient_and_skips_nonfinite
    p = Torch::NN::Parameter.new(Torch.tensor([2.0]))
    optimizer = EasyAILearning::Training::Optimizer.new({ "p" => p }, kind: :sgd, lr: 0.1)
    scaler = EasyAILearning::Training::GradScaler.new(scale: 32)
    scaler.backward(p.square.sum)

    assert scaler.step(optimizer)
    assert_in_delta 1.6, p.item, 1e-6
    optimizer.zero_grad
    scaler.backward((p * Float::INFINITY).sum)

    refute scaler.step(optimizer)
    assert_in_delta 1.6, p.item, 1e-6
    assert_in_delta 16, scaler.scale, 1e-12
  end

  def test_full_batch_dropout_and_schedule_resume_exactly
    Torch.manual_seed(8)
    model = EasyAILearning::BasicNN::Mlp.new(dropout: 0.2)
    x, target = Torch.ones([2, 2]), Torch.zeros([2, 2])
    objective = -> { (model.call(x) - target).square.mean }
    loop = EasyAILearning::Training::Loop.new(model, schedule: true, seed: 5)
    loop.run(steps: 2, total_steps: 4, &objective)
    restored = EasyAILearning::BasicNN::Mlp.new(dropout: 0.2)
    restored.load_state_dict(model.state_dict)
    resumed = EasyAILearning::Training::Loop.new(restored).load_state_dict(JSON.parse(JSON.generate(loop.state_dict)))
    loop.run(steps: 2, total_steps: 4, &objective)
    resumed.run(steps: 2, total_steps: 4) { (restored.call(x) - target).square.mean }

    model.named_parameters.each { |name, p| assert_equal p.to_a, restored.named_parameters[name].to_a }
    assert_equal loop.history, resumed.history
  end
end
