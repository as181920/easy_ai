module EasyAI
  module Modules
    class FeedForward < Torch::NN::Module
      def initialize(embed_dim:, hidden_dim:, dropout: 0.1)
        super()
        @linear1 = Torch::NN::Linear.new(embed_dim, hidden_dim)
        @linear2 = Torch::NN::Linear.new(hidden_dim, embed_dim)
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      def forward(x)
        x = EasyAI::Utils::TensorOps.gelu(@linear1.call(x))
        x = @dropout.call(x)
        @linear2.call(x)
      end
    end
  end
end
