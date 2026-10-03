module EasyAILearning
  module Training
    # Readable scalar reference, independent of Torch optimizer implementations.
    class ScalarOptimizer
      attr_reader :step, :first, :second, :velocity

      def initialize(kind:, lr: 0.1, decay: 0.0, momentum: 0.9, betas: [0.9, 0.999], epsilon: 1e-8)
        kind = kind.to_s.to_sym
        raise ArgumentError, "Unknown optimizer" unless %i[sgd momentum adam adamw].include?(kind)
        raise ArgumentError, "Invalid optimizer settings" unless lr > 0 && decay >= 0 && momentum >= 0 && momentum < 1 &&
          betas.all? { |b| b >= 0 && b < 1 } && epsilon > 0

        @kind, @lr, @decay, @momentum, @betas, @epsilon = kind, lr, decay, momentum, betas, epsilon
        @step, @first, @second, @velocity = 0, 0.0, 0.0, 0.0
      end

      def update(value, gradient)
        @step += 1
        g = @kind == :adamw ? gradient : gradient + @decay * value
        case @kind
        when :sgd then value - @lr * g
        when :momentum
          @velocity = @momentum * velocity + g
          value - @lr * velocity
        else
          b1, b2 = @betas
          @first = b1 * first + (1 - b1) * g
          @second = b2 * second + (1 - b2) * g * g
          m, v = first / (1 - b1**step), second / (1 - b2**step)
          value * (@kind == :adamw ? 1 - @lr * @decay : 1) - @lr * m / (::Math.sqrt(v) + @epsilon)
        end
      end
    end
  end
end
