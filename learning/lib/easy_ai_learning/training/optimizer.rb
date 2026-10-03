module EasyAILearning
  module Training
    # Uses the same equations as ScalarOptimizer, now on named tensors.
    class Optimizer
      attr_reader :state, :param_groups, :kind, :named

      def initialize(named, kind: :adamw, lr: 0.02, decay: 0.0, momentum: 0.9, betas: [0.9, 0.999], epsilon: 1e-8)
        kind = kind.to_s.to_sym
        raise ArgumentError, "Unknown optimizer" unless %i[sgd momentum adam adamw].include?(kind)
        raise ArgumentError, "Invalid optimizer settings" unless lr.finite? && lr > 0 && decay >= 0 &&
          momentum.between?(0, 1) && momentum < 1 && betas.size == 2 && betas.all? { |v| v >= 0 && v < 1 } && epsilon > 0
        @named, @kind = named.select { |_, p| p.requires_grad }, kind
        raise ArgumentError, "No trainable parameters" if @named.empty?
        @param_groups = [{ lr: lr, decay: decay, momentum: momentum, betas: betas, epsilon: epsilon }]
        @state = {}
      end

      def zero_grad
        named.each_value { |p| p.grad.zero! if p.grad }
      end

      def step
        settings = param_groups.first
        raise FloatDomainError, "Nonfinite gradient" unless named.values.all? { |p| !p.grad || p.grad.isfinite.all.item }
        Torch.no_grad do
          named.each do |name, p|
            next unless p.grad
            g = kind == :adamw ? p.grad : p.grad + settings[:decay] * p
            s = state[name] ||= { step: 0, first: Torch.zeros_like(p), second: Torch.zeros_like(p), velocity: Torch.zeros_like(p) }
            s[:step] += 1
            update = case kind
                     when :sgd then g
                     when :momentum
                       s[:velocity] = settings[:momentum] * s[:velocity] + g
                     else
                       b1, b2 = settings[:betas]
                       s[:first] = b1 * s[:first] + (1 - b1) * g
                       s[:second] = b2 * s[:second] + (1 - b2) * g.square
                       m = s[:first] / (1 - b1**s[:step])
                       v = s[:second] / (1 - b2**s[:step])
                       m / (v.sqrt + settings[:epsilon])
                     end
            p.mul!(1 - settings[:lr] * settings[:decay]) if kind == :adamw
            p.add!(update, alpha: -settings[:lr])
          end
        end
      end

      def state_dict
        { kind: kind.to_s, settings: param_groups.first,
          state: state.transform_values { |s| s.transform_values { |v| v.is_a?(Torch::Tensor) ? v.detach.cpu.to_a : v } } }
      end

      def load_state_dict(saved)
        saved = saved.deep_symbolize_keys
        raise ArgumentError, "Optimizer kind mismatch" unless saved.fetch(:kind) == kind.to_s
        @param_groups = [saved.fetch(:settings)]
        @state = saved.fetch(:state).to_h do |name, entries|
          p = named.fetch(name.to_s)
          [name.to_s, entries.to_h do |key, value|
            [key, key == :step ? value : Torch.tensor(value, dtype: p.dtype, device: p.device)]
          end]
        end
        self
      end
    end
  end
end
