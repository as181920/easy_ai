module EasyAILearning
  module Attention
    class Additive < Torch::NN::Module
      attr_reader :last_weights

      def initialize(hidden: 8)
        super()
        @query = Torch::NN::Linear.new(hidden, hidden, bias: false)
        @key = Torch::NN::Linear.new(hidden, hidden, bias: false)
        @score = Torch::NN::Linear.new(hidden, 1, bias: false)
      end

      def forward(query, memory)
        scores = @score.call(Torch.tanh(@query.call(query).unsqueeze(1) + @key.call(memory))).squeeze(-1)
        @last_weights = Torch::NN::Functional.softmax(scores, dim: -1)
        (memory * last_weights.unsqueeze(-1)).sum(1)
      end
    end
  end
end
