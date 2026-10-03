module EasyAILearning
  module Attention
    class MultiHead < Torch::NN::Module
      attr_reader :q_proj, :k_proj, :v_proj, :o_proj, :num_heads, :head_dim, :last_weights

      def initialize(embed_dim:, num_heads: 1, dropout: 0.0)
        super()
        raise ArgumentError, "Invalid head dimensions" unless embed_dim > 0 && num_heads > 0 && embed_dim % num_heads == 0
        @num_heads, @head_dim = num_heads, embed_dim / num_heads
        @q_proj, @k_proj, @v_proj, @o_proj = Array.new(4) { Torch::NN::Linear.new(embed_dim, embed_dim) }
        @dropout = Torch::NN::Dropout.new(p: dropout)
      end

      def forward(query, memory: nil, padding_mask: nil, causal: false)
        memory ||= query
        batch, qlength, = query.shape
        klength = memory.shape[1]
        q, k, v = reshape(q_proj.call(query)), reshape(k_proj.call(memory)), reshape(v_proj.call(memory))
        scores = Torch.matmul(q, k.transpose(-2, -1)) / ::Math.sqrt(head_dim)
        valid = Torch.ones([batch, 1, qlength, klength], dtype: :bool, device: query.device)
        if padding_mask
          raise ArgumentError, "Padding mask must be [batch, key length]" unless padding_mask.shape == [batch, klength]
          valid = valid.logical_and(padding_mask.to(dtype: :bool).unsqueeze(1).unsqueeze(1))
        end
        if causal
          raise ArgumentError, "Causal self-attention requires equal lengths" unless qlength == klength
          valid = valid.logical_and(Utils::TensorOps.causal_mask(qlength, device: query.device).unsqueeze(0).unsqueeze(0))
        end
        raise ArgumentError, "A query has no visible keys" unless valid.any(-1).all.item
        @last_weights = Torch::NN::Functional.softmax(scores.masked_fill(valid.logical_not, -Float::INFINITY), dim: -1)
        weighted = Torch.matmul(@dropout.call(last_weights), v)
        o_proj.call(weighted.transpose(1, 2).contiguous.reshape([batch, qlength, num_heads * head_dim]))
      end

      private

      def reshape(x)
        x.reshape([x.shape[0], x.shape[1], num_heads, head_dim]).transpose(1, 2)
      end
    end
  end
end
