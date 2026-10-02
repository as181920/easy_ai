module EasyAILearning
  module BasicNN
    # Dense(2, 2) -> ReLU -> Dense(2, 4), with explicit scalar backpropagation.
    class ScalarLogicNetwork
      attr_reader :parameters

      def initialize(seed: 1337)
        rng = Random.new(seed)
        @parameters = {
          hidden_weights: Array.new(2) { Array.new(2) { rng.rand(0.4..0.8) } },
          hidden_biases: [rng.rand(0.0..0.2), rng.rand(-0.6..-0.4)],
          output_weights: Array.new(4) { Array.new(2) { rng.rand(-0.5..0.5) } },
          output_biases: Array.new(4) { rng.rand(-0.1..0.1) }
        }
      end

      # A separate existence proof, never used to initialize the training run.
      def self.exact
        model = new
        model.parameters.replace(hidden_weights: [[1.0, 1.0], [1.0, 1.0]], hidden_biases: [0.0, -1.0],
          output_weights: [[0.0, 1.0], [1.0, -1.0], [0.0, -1.0], [1.0, -2.0]], output_biases: [0.0, 0.0, 1.0, 0.0])
        model
      end

      def forward(input)
        hidden = hidden_values(input).map { |value| relu(value) }
        affine(hidden, parameters[:output_weights], parameters[:output_biases])
      end

      def relu(value)
        [0.0, value].max
      end

      def parameter_count
        parameters.values.sum { |values| values.flatten.size }
      end

      def loss(inputs, targets)
        validate_batch!(inputs, targets)
        inputs.each_with_index.sum do |input, row|
          forward(input).each_with_index.sum { |score, gate| (score - targets[row][gate])**2 }
        end / (inputs.size * 4.0)
      end

      def gradients(inputs, targets)
        validate_batch!(inputs, targets)
        result = zero_gradients
        inputs.each_with_index do |input, row|
          accumulate_gradients(input, targets[row], result, scale: 2.0 / (inputs.size * 4))
        end
        result
      end

      def update!(gradients, learning_rate:)
        raise ArgumentError, "Learning rate must be finite and positive" unless learning_rate.finite? && learning_rate > 0
        parameters.each do |name, values|
          values.each_index do |index|
            if values[index].is_a?(Array)
              values[index].each_index { |column| values[index][column] -= learning_rate * gradients[name][index][column] }
            else
              values[index] -= learning_rate * gradients[name][index]
            end
          end
        end
      end

      private

      def hidden_values(input)
        raise ArgumentError, "Expected two finite numbers" unless input.size == 2 && input.all? { |value| value.is_a?(Numeric) && value.finite? }
        affine(input, parameters[:hidden_weights], parameters[:hidden_biases])
      end

      def affine(input, weights, biases)
        weights.each_with_index.map { |row, index| row.each_with_index.sum { |weight, column| weight * input[column] } + biases[index] }
      end

      def zero_gradients
        parameters.transform_values { |values| values.map { |value| value.is_a?(Array) ? Array.new(value.size, 0.0) : 0.0 } }
      end

      def accumulate_gradients(input, target, result, scale:)
        preactivation = hidden_values(input)
        hidden = preactivation.map { |value| relu(value) }
        output = affine(hidden, parameters[:output_weights], parameters[:output_biases])
        output.each_index do |gate|
          # d(mean squared error)/d(output) = 2 * (output - target) / 16.
          derivative = scale * (output[gate] - target[gate])
          result[:output_biases][gate] += derivative
          hidden.each_index do |neuron|
            result[:output_weights][gate][neuron] += derivative * hidden[neuron]
            accumulate_hidden_gradient(input, neuron, preactivation[neuron], derivative * parameters[:output_weights][gate][neuron], result)
          end
        end
      end

      def accumulate_hidden_gradient(input, neuron, preactivation, derivative, result)
        return unless preactivation > 0 # ReLU derivative; choose zero at the kink.
        result[:hidden_biases][neuron] += derivative
        input.each_index { |column| result[:hidden_weights][neuron][column] += derivative * input[column] }
      end

      def validate_batch!(inputs, targets)
        unless !inputs.empty? && inputs.size == targets.size && targets.all? { |row| row.size == 4 && row.all? { |value| value.is_a?(Numeric) && value.finite? } }
          raise ArgumentError, "Expected nonempty inputs with four finite targets per row"
        end
      end
    end
  end
end
