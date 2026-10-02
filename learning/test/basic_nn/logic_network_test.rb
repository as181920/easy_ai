require_relative "../test_helper"
require "tmpdir"

class LogicNetworkTest < Minitest::Test
  def test_branches_perceptrons_and_exact_relu_agree_on_the_complete_truth_table
    gates = EasyAILearning::BasicNN::LogicGates
    targets = [[0, 0, 1, 0], [0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]]

    assert_equal [[0], [1], [1], [0]], gates.targets
    gates::INPUTS.each_with_index do |input, row|
      assert_equal [targets[row].last], EasyAILearning::BasicNN::LogicNetwork.exact.scores(input)
      assert_equal targets[row].first(3), %w[and or nand].map { |gate| gates.perceptron(gate, *input) }
    end
    assert_raises(ArgumentError) { gates.perceptron(:xor, 0, 1) }
  end

  def test_manual_gradients_match_finite_differences_for_every_parameter
    model = EasyAILearning::BasicNN::ScalarLogicNetwork.new
    inputs, targets = EasyAILearning::BasicNN::LogicGates::INPUTS, EasyAILearning::BasicNN::LogicGates.targets
    gradients = model.gradients(inputs, targets)
    model.parameters.each do |name, values|
      values.each_index do |row|
        if values[row].is_a?(Array)
          values[row].each_index { |column| check_derivative(model, values[row], column, gradients[name][row][column], inputs, targets) }
        else
          check_derivative(model, values, row, gradients[name][row], inputs, targets)
        end
      end
    end
  end

  def test_randomly_initialized_network_learns_xor_without_exact_weights
    model = EasyAILearning::BasicNN::LogicNetwork.new(seed: 1337, device: :cpu)
    trainer = EasyAILearning::BasicNN::LogicTrainer.new(model: model).train

    assert_equal 9, model.parameter_count
    assert_predicate trainer, :converged?
    assert_operator trainer.history.last.last, :<, trainer.history.first.last
    actual = EasyAILearning::BasicNN::LogicGates::INPUTS.map { |input| model.scores(input).map { |value| value >= 0.5 ? 1 : 0 } }

    assert_equal EasyAILearning::BasicNN::LogicGates.targets, actual
    assert_equal [0, 1, 1, 0], EasyAILearning::BasicNN::LogicGates::INPUTS.map { |input| model.predict(input) }
  end

  def test_torch_autograd_matches_the_scalar_chain_rule
    scalar = EasyAILearning::BasicNN::ScalarLogicNetwork.new
    model = EasyAILearning::BasicNN::LogicNetwork.new(device: :cpu)
    inputs, targets = EasyAILearning::BasicNN::LogicGates::INPUTS, EasyAILearning::BasicNN::LogicGates.targets
    loss = Torch::NN::Functional.mse_loss(model.call(Torch.tensor(inputs, dtype: :float32)), Torch.tensor(targets, dtype: :float32))
    loss.backward
    reference, actual = scalar.gradients(inputs, targets), model.parameter_gradients
    reference.each do |name, values|
      values.flatten.zip(actual.fetch(name).flatten).each { |expected, value| assert_in_delta expected, value, 1e-6 }
    end
  end

  def test_saved_xor_parameters_reload_without_training
    model = EasyAILearning::BasicNN::LogicNetwork.exact(device: :cpu)
    Dir.mktmpdir do |directory|
      path = File.join(directory, "model.json")
      File.write(path, JSON.generate(architecture: [2, 2, 1], activation: "ReLU", outputs: ["xor"], parameters: model.parameter_values))
      restored = EasyAILearning::BasicNN::LogicNetwork.load(path, device: "cpu")

      assert_equal model.parameter_values, restored.parameter_values
      assert_equal [0, 1, 1, 0], EasyAILearning::BasicNN::LogicGates::INPUTS.map { |input| restored.predict(input) }
      assert_in_delta 1.0, restored.score([1, 0]), 1e-7
    end
  end

  private

  def check_derivative(model, values, index, expected, inputs, targets)
    original, epsilon = values[index], 1e-6
    values[index] = original + epsilon
    plus = model.loss(inputs, targets)
    values[index] = original - epsilon
    minus = model.loss(inputs, targets)

    assert_in_delta expected, (plus - minus) / (2 * epsilon), 1e-7
  ensure
    values[index] = original
  end
end
