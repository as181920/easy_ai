module EasyAI
  module NN
    class FeedForward < Torch::NN::Module
      attr_reader :up, :down

      def initialize(hidden_size:, intermediate_size:, dropout: 0.0)
        super()
        @up = Torch::NN::Linear.new(hidden_size, intermediate_size)
        @down = Torch::NN::Linear.new(intermediate_size, hidden_size)
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      def forward(x)
        @down.call(@dropout.call(Torch::NN::Functional.gelu(@up.call(x))))
      end
    end
  end
end
