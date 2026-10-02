require_relative "../../../../lib/easy_ai"
require_relative "scalar_logic_network"
require "json"

module EasyAILearning
  module BasicNN
    # Torch computes tensors and automatic gradients; Ruby defines the architecture.
    class LogicNetwork < Torch::NN::Module
      attr_reader :device

      def initialize(seed: 1337, device: :auto)
        super()
        @hidden = Torch::NN::Linear.new(2, 2)
        @output = Torch::NN::Linear.new(2, 1)
        @device = EasyAI::Runtime::DevicePolicy.new(requested: device).resolve
        to(@device) # Place parameters before constructing the optimizer.
        set_parameters(ScalarLogicNetwork.new(seed: seed).parameters)
      end

      def self.exact(device: :cpu)
        new(device: device).tap { |model| model.set_parameters(ScalarLogicNetwork.exact.parameters) }
      end

      def self.load(path, device: :auto)
        saved = JSON.parse(File.read(path))
        unless saved.fetch("architecture") == [2, 2, 1] && saved.fetch("activation") == "ReLU" && saved.fetch("outputs") == %w[xor]
          raise ArgumentError, "Expected a 2 -> 2 ReLU -> 1 XOR model"
        end
        new(device: device).tap { |model| model.set_parameters(saved.fetch("parameters")) }.eval
      end

      def forward(input)
        @output.call(Torch::NN::Functional.relu(@hidden.call(input)))
      end

      def scores(input)
        batch_scores([input]).first
      end

      def score(input)
        scores(input).first
      end

      def predict(input)
        score(input) >= 0.5 ? 1 : 0
      end

      def batch_scores(inputs)
        Torch.no_grad { forward(Torch.tensor(inputs, dtype: :float32, device: device)).cpu.to_a }
      end

      def relu(value)
        Torch.relu(Torch.tensor([value], dtype: :float32, device: device)).item
      end

      def parameter_count
        parameters.sum(&:numel)
      end

      def parameter_values
        { hidden_weights: @hidden.weight.detach.cpu.to_a, hidden_biases: @hidden.bias.detach.cpu.to_a,
          output_weights: @output.weight.detach.cpu.to_a, output_biases: @output.bias.detach.cpu.to_a }
      end

      def parameter_gradients
        { hidden_weights: @hidden.weight.grad.cpu.to_a, hidden_biases: @hidden.bias.grad.cpu.to_a,
          output_weights: @output.weight.grad.cpu.to_a, output_biases: @output.bias.grad.cpu.to_a }
      end

      def set_parameters(values)
        values = values.transform_keys(&:to_sym)
        tensors = { hidden_weights: @hidden.weight, hidden_biases: @hidden.bias,
          output_weights: @output.weight, output_biases: @output.bias }
        Torch.no_grad do
          tensors.each { |name, tensor| tensor.copy!(Torch.tensor(values.fetch(name), dtype: :float32, device: device)) }
        end
      end
    end
  end
end
