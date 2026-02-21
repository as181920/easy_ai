module EasyAI
  module Modules
    class TransformerBlock < Torch::NN::Module
      def initialize(embed_dim:, num_heads:, dropout: 0.1)
        super()
        @ln1 = Torch::NN::LayerNorm.new(embed_dim)
        @attn = Attention.new(embed_dim: embed_dim, num_heads: num_heads, dropout: dropout)
        @ln2 = Torch::NN::LayerNorm.new(embed_dim)
        ff_hidden = embed_dim * 4
        @ff = FeedForward.new(embed_dim: embed_dim, hidden_dim: ff_hidden, dropout: dropout)
        @dropout = Torch::NN::Dropout.new(dropout)
      end

      def forward(x)
        attn_output = @attn.call(@ln1.call(x))
        x = x + @dropout.call(attn_output)
        ff_output = @ff.call(@ln2.call(x))
        x + @dropout.call(ff_output)
      end
    end
  end
end
