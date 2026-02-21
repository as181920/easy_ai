module EasyAI
  module Modules
    class PositionalEmbeddings < Torch::NN::Module
      def initialize(block_size:, embedding_dim:)
        super()
        @positional_embeddings = Torch::NN::Embedding.new(block_size, embedding_dim)
      end

      def forward(x)
        batch_size, seq_len = x.shape
        positions = Torch.arange(0, seq_len, dtype: :int64, device: x.device)
        pos_emb = @positional_embeddings.call(positions)
        pos_emb.unsqueeze(0).expand(batch_size, seq_len, -1)
      end
    end
  end
end
