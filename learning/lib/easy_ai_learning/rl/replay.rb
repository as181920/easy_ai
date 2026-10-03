module EasyAILearning
  module RL
    class Replay
      attr_reader :entries

      def initialize(capacity: 128, seed: 1337)
        raise ArgumentError, "Positive capacity required" unless capacity > 0
        @capacity, @rng, @entries = capacity, Random.new(seed), []
      end

      def push(transition)
        entries << transition.dup
        entries.shift if entries.size > @capacity
      end

      def sample(size)
        raise ArgumentError, "Invalid sample size" unless size.between?(1, entries.size)
        entries.sample(size, random: @rng)
      end
    end
  end
end
