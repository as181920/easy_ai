require_relative "../test_helper"
require "tmpdir"

class TrainingTest < Minitest::Test
  def test_optimizer_equations_match_independent_scalar_reference
    %i[sgd momentum adam adamw].each do |kind|
      parameter = Torch::NN::Parameter.new(Torch.tensor([2.0]))
      optimizer = EasyAILearning::Training::Optimizer.new({ "p" => parameter }, kind: kind, lr: 0.1, decay: 0.2)
      reference = EasyAILearning::Training::ScalarOptimizer.new(kind: kind, lr: 0.1, decay: 0.2)
      value = 2.0
      [0.5, -0.3, 0.0].each do |gradient|
        optimizer.zero_grad
        (parameter * gradient).sum.backward
        optimizer.step
        value = reference.update(value, gradient)

        assert_in_delta value, parameter.item, 1e-6
      end
    end
  end

  def test_optimizer_resume_matches_next_update
    p = Torch::NN::Parameter.new(Torch.tensor([1.0]))
    first = EasyAILearning::Training::Optimizer.new({ "p" => p })
    p.square.sum.backward
    first.step
    q = Torch::NN::Parameter.new(p.detach.clone)
    second = EasyAILearning::Training::Optimizer.new({ "p" => q }).load_state_dict(first.state_dict)
    first.zero_grad
    p.square.sum.backward
    q.square.sum.backward
    first.step
    second.step

    assert_in_delta p.item, q.item, 1e-7
  end

  def test_dropout_and_schedule
    math = EasyAILearning::Training::Math
    x = Torch.tensor([2.0, 4.0])

    assert_equal [4.0, 0.0], math.dropout(x, probability: 0.5, mask: Torch.tensor([true, false])).to_a
    assert_equal x.to_a, math.dropout(x, probability: 0.5, training: false).to_a
    assert_in_delta 0.5, math.learning_rate(0, total: 10, base: 1, warmup: 2), 1e-12
    assert_in_delta 0, math.learning_rate(9, total: 10, base: 1, warmup: 2), 1e-12
  end

  def test_gradient_clipping_preserves_direction
    math = EasyAILearning::Training::Math
    p = Torch::NN::Parameter.new(Torch.tensor([0.0, 0.0]))
    (p * Torch.tensor([3.0, 4.0])).sum.backward

    assert_in_delta 5, math.clipped_gradients([p], 1)[:before], 1e-7
    assert_in_delta 1, p.grad.norm.item, 1e-7
  end

  def test_loss_mask_and_normalization
    logits = Torch.tensor([[[0.0, 0.0], [10.0, -10.0]]])
    mask = Torch.tensor([[true, false]])
    math = EasyAILearning::Training::Math

    assert_in_delta Math.log(2), math.masked_cross_entropy(logits, Torch.tensor([[1, 1]], dtype: :int64), mask: mask).item, 1e-6
    assert_raises(ArgumentError) { math.masked_cross_entropy(logits, Torch.tensor([[1, 0]], dtype: :int64), mask: Torch.zeros([1, 2], dtype: :bool)) }
    stats = math.standardize_fit([[0.0], [2.0]])

    assert_equal [[-1.0], [1.0]], math.standardize([[0.0], [2.0]], stats)
    assert_equal [[99.0]], math.standardize([[100.0]], stats)
  end

  def test_statistics_and_artifact_roundtrip
    stats = EasyAILearning::Diagnostics::Stats.summarize([-2, 0, 2])

    assert_in_delta Math.sqrt(8.0 / 3), stats[:rms], 1e-12
    assert_equal 0, stats[:median]
    model = EasyAILearning::BasicNN::Mlp.new
    Dir.mktmpdir do |directory|
      artifacts = EasyAILearning::Course::Artifacts.new(directory)
      artifacts.save_model("model", model)
      restored = EasyAILearning::BasicNN::Mlp.new
      EasyAILearning::Course::Artifacts.load_model(File.join(directory, "model.json"), restored)
      x = Torch.tensor([[0.5, -0.2]])

      assert_equal model.eval.call(x).to_a, restored.call(x).to_a
    end
  end
end

class InvalidTrainingInputsTest < Minitest::Test
  def test_masked_out_of_vocabulary_targets_are_not_evaluated
    logits = Torch::NN::Parameter.new(Torch.zeros([1, 2, 2]))
    labels = Torch.tensor([[0, 999]], dtype: :int64)
    loss = EasyAILearning::Training::Math.masked_cross_entropy(logits, labels, mask: Torch.tensor([[true, false]]))
    loss.backward

    assert_in_delta Math.log(2), loss.item, 1e-6
    assert_equal [0.0, 0.0], logits.grad[0][1].to_a
  end

  def test_nonfinite_gradients_are_rejected_before_updates
    parameter = Torch::NN::Parameter.new(Torch.tensor([1.0]))
    optimizer = EasyAILearning::Training::Optimizer.new({ "p" => parameter })
    (parameter * Float::INFINITY).sum.backward

    assert_raises(FloatDomainError) { optimizer.step }
    assert_raises(FloatDomainError) { EasyAILearning::Training::Math.clipped_gradients([parameter], 1) }
    assert_in_delta 1, parameter.item, 1e-12
  end

  def test_wrong_artifact_shape_is_rejected
    model = EasyAILearning::BasicNN::Mlp.new(hidden: 3)
    Dir.mktmpdir do |directory|
      artifacts = EasyAILearning::Course::Artifacts.new(directory)
      artifacts.save_model("model", model)
      path = File.join(directory, "model.json")

      assert_raises(ArgumentError) { EasyAILearning::Course::Artifacts.load_model(path, EasyAILearning::BasicNN::Mlp.new(hidden: 4)) }
    end
  end
end
