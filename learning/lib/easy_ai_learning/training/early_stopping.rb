module EasyAILearning
  module Training
    class EarlyStopping
      attr_reader :best, :bad_steps, :best_state

      def initialize(patience: 5, minimum_delta: 0.0)
        raise ArgumentError, "Invalid stopping settings" unless patience > 0 && minimum_delta >= 0
        @patience, @delta, @best, @bad_steps = patience, minimum_delta, Float::INFINITY, 0
      end

      def observe(loss, model)
        raise ArgumentError, "Finite validation loss required" unless loss.finite?
        if loss < best - @delta
          @best, @bad_steps = loss, 0
          @best_state = model.state_dict.transform_values { |v| v.detach.clone }
        else
          @bad_steps += 1
        end
        bad_steps >= @patience
      end

      def restore(model)
        raise ArgumentError, "No observation" unless best_state
        model.load_state_dict(best_state)
      end
    end
  end
end
