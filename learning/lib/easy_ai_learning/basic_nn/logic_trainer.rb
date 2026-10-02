module EasyAILearning
  module BasicNN
    class LogicTrainer
      attr_reader :model, :history, :steps, :learning_rate, :max_steps, :tolerance

      def initialize(model:, learning_rate: 0.1, max_steps: 10_000, tolerance: 0.01)
        raise ArgumentError, "Steps must be positive" unless max_steps.is_a?(Integer) && max_steps > 0
        raise ArgumentError, "Tolerance must be between 0 and 0.5" unless tolerance > 0 && tolerance < 0.5
        raise ArgumentError, "Learning rate must be finite and positive" unless learning_rate.finite? && learning_rate > 0
        @model, @learning_rate, @max_steps, @tolerance = model, learning_rate, max_steps, tolerance
        @history, @steps = [], 0
      end

      def train
        inputs = Torch.tensor(LogicGates::INPUTS, dtype: :float32, device: model.device)
        targets = Torch.tensor(LogicGates.targets, dtype: :float32, device: model.device)
        optimizer = Torch::Optim::SGD.new(model.parameters, lr: learning_rate)
        @history = [[0, measure(inputs, targets)]]
        model.train
        @max_steps.times do |index|
          update(inputs, targets, optimizer)
          @steps = index + 1
          value = measure(inputs, targets)
          raise FloatDomainError, "Training diverged" unless value.finite?
          history << [steps, value]
          yield(steps, value) if block_given?
          GC.start if (steps % 100).zero?
          break if converged?
        end
        model.eval
        self
      end

      def max_error
        @max_error || Float::INFINITY
      end

      def converged?
        max_error <= @tolerance
      end

      private

      def update(inputs, targets, optimizer)
        optimizer.zero_grad
        loss = Torch::NN::Functional.mse_loss(model.call(inputs), targets)
        loss.backward
        optimizer.step
      end

      def measure(inputs, targets)
        Torch.no_grad do
          prediction = model.call(inputs)
          @max_error = (prediction - targets).abs.max.item
          Torch::NN::Functional.mse_loss(prediction, targets).item
        end
      end
    end
  end
end
