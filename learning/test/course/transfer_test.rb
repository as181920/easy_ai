require_relative "../test_helper"

class TransferTest < Minitest::Test
  def test_lora_starts_as_base_and_merged_weight_matches
    base = Torch::NN::Linear.new(2, 3)
    layer = EasyAILearning::Transfer::LowRankLinear.new(base, rank: 1, alpha: 2)
    x = Torch.tensor([[1.0, 2.0]])

    assert_equal base.call(x).to_a, layer.call(x).to_a
    Torch.no_grad { layer.b.weight.fill!(0.5) }
    expected = Torch.matmul(x, layer.merged_weight.transpose(0, 1)) + base.bias

    assert_in_delta 0, (expected - layer.call(x)).abs.max.item, 1e-6
    layer.call(x).sum.backward

    assert base.parameters.all? { |p| !p.requires_grad }
    assert_operator layer.b.weight.grad.numel, :>, 0
  end

  def test_freeze_prevents_update_without_asserting_training_effect
    model = EasyAILearning::BasicNN::Mlp.new
    model.hidden.parameters.each { |p| p.requires_grad = false }
    before = model.hidden.weight.clone
    optimizer = EasyAILearning::Training::Optimizer.new(model.named_parameters)
    model.call(Torch.ones([2, 2])).square.mean.backward
    optimizer.step

    assert_equal before.to_a, model.hidden.weight.to_a
    assert_equal %w[output.weight output.bias].sort, optimizer.named.keys.sort
  end

  def test_distillation_detaches_teacher_and_matches_uniform_soft_target
    student = Torch::NN::Parameter.new(Torch.zeros([1, 2]))
    teacher = Torch::NN::Parameter.new(Torch.zeros([1, 2]))
    loss = EasyAILearning::Transfer::Distillation.loss(student, teacher, Torch.tensor([0], dtype: :int64), temperature: 2, soft_weight: 1)

    assert_in_delta 4 * Math.log(2), loss.item, 1e-6
    loss.backward

    assert teacher.grad.nil? || teacher.grad.numel.zero?
    assert_equal [[0.0, 0.0]], student.grad.to_a
  end
end

class AdaptedModelTest < Minitest::Test
  def test_low_rank_mlp_only_registers_adapter_as_trainable
    model = EasyAILearning::Transfer::AdaptedMlp.new(rank: 1)
    optimizer = EasyAILearning::Training::Optimizer.new(model.named_parameters)

    assert_equal %w[output.a.weight output.b.weight].sort, optimizer.named.keys.sort
    assert_equal [2, 2], model.call(Torch.ones([2, 2])).shape
  end
end
