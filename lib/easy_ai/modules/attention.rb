module EasyAI
  module Modules
    class Attention < Torch::NN::Module
      attr_reader :num_heads, :head_dim

      def initialize(embed_dim:, num_heads:, dropout: 0.1)
        super()
        raise ArgumentError, "embed_dim must be divisible by num_heads" unless (embed_dim % num_heads).zero?

        @num_heads = num_heads
        @head_dim = embed_dim / num_heads

        @q_proj = Torch::NN::Linear.new(embed_dim, embed_dim)
        @k_proj = Torch::NN::Linear.new(embed_dim, embed_dim)
        @v_proj = Torch::NN::Linear.new(embed_dim, embed_dim)
        @o_proj = Torch::NN::Linear.new(embed_dim, embed_dim)
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      def forward(x)
        batch_size, seq_len, _ = x.shape

        q = reshape_heads(@q_proj.call(x))
        k = reshape_heads(@k_proj.call(x))
        v = reshape_heads(@v_proj.call(x))

        attn_scores = Torch.matmul(q, k.transpose(-2, -1)) / Math.sqrt(head_dim)
        mask = EasyAI::Utils::TensorOps.causal_mask(seq_len, device: x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(Torch.logical_not(mask), -1e9)

        attn_weights = Torch::NN::Functional.softmax(attn_scores, dim: -1)
        attn_weights = @dropout.call(attn_weights)

        attn_output = Torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous.view([batch_size, seq_len, num_heads * head_dim])
        @o_proj.call(attn_output)
      end

      private

        def reshape_heads(tensor)
          batch_size, seq_len, embed_dim = tensor.shape
          tensor.view([batch_size, seq_len, num_heads, head_dim]).transpose(1, 2)
        end
    end
  end
end
