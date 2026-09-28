module EasyAI
  module Optim
    # Named state is independent of Ruby tensor object identity, allowing device
    # migration, architecture growth and exact same-device checkpoint resume.
    class AdamW
      attr_accessor :learning_rate
      attr_reader :steps

      def initialize(named_parameters, learning_rate:, weight_decay: 0.01, betas: [0.9, 0.999], epsilon: 1e-8)
        @parameters = named_parameters
        @learning_rate, @weight_decay, @betas, @epsilon = learning_rate, weight_decay, betas, epsilon
        @moments, @steps = {}, {}
      end

      def zero_grad
        @parameters.each_value { |p| p.grad&.detach!&.zero! }
      end

      def clip_grad_norm!(limit)
        grads = @parameters.values.filter_map(&:grad)
        return 0.0 if grads.empty?
        norm = Torch.stack(grads.map { |grad| grad.detach.square.sum }).sum.sqrt.item
        raise FloatDomainError, "Non-finite gradient norm" unless norm.finite?
        if norm > limit
          Torch.no_grad { grads.each { |grad| grad.mul!(limit / (norm + 1e-6)) } }
        end
        norm
      end

      def step
        Torch.no_grad do
          @parameters.each do |name, parameter|
            grad = parameter.grad
            next unless grad
            m, v = @moments[name] ||= [Torch.zeros_like(parameter), Torch.zeros_like(parameter)]
            t = @steps[name] = @steps.fetch(name, 0) + 1
            b1, b2 = @betas
            m.mul!(b1).add!(grad, alpha: 1 - b1)
            v.mul!(b2).addcmul!(grad, grad, value: 1 - b2)
            denominator = v.sqrt / Math.sqrt(1 - b2**t) + @epsilon
            parameter.mul!(1 - learning_rate * @weight_decay)
            parameter.addcdiv!(m, denominator, value: -learning_rate / (1 - b1**t))
          end
        end
      end

      def state_dict
        tensors = {}
        @moments.each do |name, (m, v)|
          tensors["#{name}/m"] = m.detach.cpu.clone
          tensors["#{name}/v"] = v.detach.cpu.clone
        end
        { "steps" => steps.dup, "tensors" => tensors, "learning_rate" => learning_rate,
         "weight_decay" => @weight_decay, "betas" => @betas, "epsilon" => @epsilon }
      end

      def load_state_dict(state, allow_growth: false)
        @learning_rate = state.fetch("learning_rate")
        @weight_decay, @betas, @epsilon = state.values_at("weight_decay", "betas", "epsilon")
        unknown = state.fetch("steps").keys - @parameters.keys
        raise ArgumentError, "Optimizer has unknown parameters: #{unknown}" unless unknown.empty?
        @moments, @steps = {}, {}
        state.fetch("steps").each do |name, step|
          parameter = @parameters.fetch(name)
          @steps[name] = step
          @moments[name] = %w[m v].map do |moment|
            old = state.fetch("tensors").fetch("#{name}/#{moment}")
            if old.shape == parameter.shape
              old.to(parameter.device).clone
            elsif allow_growth && old.shape.length == parameter.shape.length && old.shape.zip(parameter.shape).all? { |a, b| a <= b }
              enlarged = Torch.zeros_like(parameter)
              view = enlarged
              old.shape.each_with_index { |size, dimension| view = view.narrow(dimension, 0, size) }
              view.copy!(old.to(parameter.device))
              enlarged
            else
              raise ArgumentError, "Optimizer shape mismatch for #{name}"
            end
          end
        end
        self
      end
    end
  end
end
