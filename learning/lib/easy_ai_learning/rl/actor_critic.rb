module EasyAILearning
  module RL
    class ActorCritic < Torch::NN::Module
      attr_reader :policy, :value

      def initialize(states: 5)
        super()
        @policy = BasicNN::Mlp.new(input: states, hidden: 12, output: 2, activation: :tanh)
        @value = BasicNN::Mlp.new(input: states, hidden: 12, output: 1, activation: :tanh)
      end

      def forward(x)
        [policy.call(x), value.call(x).squeeze(-1)]
      end
    end
  end
end
