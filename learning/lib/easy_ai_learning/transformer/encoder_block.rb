module EasyAILearning
  module Transformer
    class EncoderBlock < Torch::NN::Module
      def initialize(width: 8, heads: 2, dropout: 0.0, residual: true, normalize: true, post_norm: false)
        super()
        @residual, @post_norm = residual, post_norm
        @ln1 = normalize ? Torch::NN::LayerNorm.new(width) : Torch::NN::Identity.new
        @ln2 = normalize ? Torch::NN::LayerNorm.new(width) : Torch::NN::Identity.new
        @attention = Attention::MultiHead.new(embed_dim: width, num_heads: heads, dropout: dropout)
        @ff = FeedForward.new(embed_dim: width, hidden_dim: width * 2, dropout: dropout)
      end

      def forward(x, padding_mask: nil)
        if @post_norm
          a = @attention.call(x, padding_mask: padding_mask)
          x = @ln1.call(@residual ? x + a : a)
          f = @ff.call(x)
          @ln2.call(@residual ? x + f : f)
        else
          a = @attention.call(@ln1.call(x), padding_mask: padding_mask)
          x = @residual ? x + a : a
          f = @ff.call(@ln2.call(x))
          @residual ? x + f : f
        end
      end
    end
  end
end
