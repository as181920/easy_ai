module EasyAILearning
  module RL
    class Chain
      attr_reader :size, :state, :steps, :limit

      def initialize(size: 5, limit: 12)
        raise ArgumentError, "Invalid chain" unless size > 1 && limit > 0
        @size, @limit = size, limit
        reset
      end

      def reset(start: 0)
        raise ArgumentError, "Invalid start" unless start.between?(0, size - 2)
        @state, @steps = start, 0
        state
      end

      def transition(state, action)
        raise ArgumentError, "Invalid state/action" unless state.between?(0, size - 1) && [0, 1].include?(action)
        return [state, 0.0, true] if state == size - 1
        next_state = [[state + (action.zero? ? -1 : 1), 0].max, size - 1].min
        terminal = next_state == size - 1
        [next_state, terminal ? 1.0 : -0.02, terminal]
      end

      def step(action)
        raise ArgumentError, "Episode finished; reset first" if state == size - 1 || steps >= limit
        @state, reward, terminal = transition(state, action)
        @steps += 1
        { state: state, reward: reward, terminated: terminal, truncated: !terminal && steps >= limit }
      end

      def self.one_hot(states, size:, device: Torch.device("cpu"))
        Torch.tensor(states.map { |s| Array.new(size) { |i| i == s ? 1.0 : 0.0 } }, device: device)
      end
    end
  end
end
