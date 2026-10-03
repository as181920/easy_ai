require_relative "../test_helper"

class CourseOptionsTest < Minitest::Test
  def test_string_and_symbol_optimizer_kinds_have_identical_updates
    %i[sgd momentum adam adamw].each do |kind|
      a, b = Torch::NN::Parameter.new(Torch.tensor([1.0])), Torch::NN::Parameter.new(Torch.tensor([1.0]))
      first = EasyAILearning::Training::Optimizer.new({ "p" => a }, kind: kind)
      second = EasyAILearning::Training::Optimizer.new({ "p" => b }, kind: kind.to_s)
      a.square.sum.backward
      b.square.sum.backward
      first.step
      second.step

      assert_equal a.to_a, b.to_a
      scalar = EasyAILearning::Training::ScalarOptimizer

      assert_equal scalar.new(kind: kind).update(1, 2), scalar.new(kind: kind.to_s).update(1, 2)
    end
  end

  def test_string_and_symbol_cell_and_activation_options_are_equivalent
    %i[rnn lstm gru].each do |kind|
      Torch.manual_seed(8)
      first = EasyAILearning::RNN::Cell.new(input: 2, kind: kind)
      Torch.manual_seed(8)
      second = EasyAILearning::RNN::Cell.new(input: 2, kind: kind.to_s)
      x = Torch.ones([1, 2])
      a = first.call(x, first.initial(1))
      b = second.call(x, second.initial(1))
      a, b = a.first, b.first if a.is_a?(Array)

      assert_equal a.to_a, b.to_a
    end
    Torch.manual_seed(8)
    first = EasyAILearning::BasicNN::Mlp.new(activation: :tanh)
    Torch.manual_seed(8)
    second = EasyAILearning::BasicNN::Mlp.new(activation: "tanh")

    assert_equal first.call(Torch.ones([1, 2])).to_a, second.call(Torch.ones([1, 2])).to_a
    assert_raises(ArgumentError) { EasyAILearning::BasicNN::Mlp.new(activation: "unknown") }
  end

  def test_string_ensemble_kind_and_device_options
    rows, labels = [[0], [1]], [0, 1]
    ensemble = EasyAILearning::Foundations::Ensemble
    first, second = ensemble.new.fit(rows, labels, kind: :boosting), ensemble.new.fit(rows, labels, kind: "boosting")

    assert_equal first.predict([0]), second.predict([0])
    assert_equal EasyAI::Runtime::DevicePolicy.new(requested: :cpu).resolve, EasyAI::Runtime::DevicePolicy.new(requested: "cpu").resolve
    assert_raises(ArgumentError) { EasyAILearning::RNN::Cell.new(input: 2, kind: "unknown") }
  end
end
