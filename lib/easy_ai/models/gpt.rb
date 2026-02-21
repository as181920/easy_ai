module EasyAI
  module Models
    class GPT < Torch::NN::Module
      attr_reader :config

      def initialize(config)
        super()
        @config = config
        vocab_size = config[:vocab_size]
        block_size = config[:block_size]
        n_layer = config[:n_layer]
        n_head = config[:n_head]
        n_embd = config[:n_embd]
        dropout = config[:dropout]

        @token_embedding = Torch::NN::Embedding.new(vocab_size, n_embd)
        @position_embedding = EasyAI::Modules::PositionalEmbeddings.new(
          block_size: block_size,
          embedding_dim: n_embd
        )

        @dropout = Torch::NN::Dropout.new(p: dropout)
        transformer_blocks = Array.new(n_layer) do
          EasyAI::Modules::TransformerBlock.new(embed_dim: n_embd, num_heads: n_head, dropout: dropout)
        end
        @blocks = Torch::NN::ModuleList.new(transformer_blocks)
        @ln_f = Torch::NN::LayerNorm.new(n_embd)
        @lm_head = Torch::NN::Linear.new(n_embd, vocab_size, bias: false)
      end

      def forward(idx)
        batch_size, seq_len = idx.shape
        raise ArgumentError, "Sequence length exceeds block size" if seq_len > config[:block_size]

        token_embeddings = @token_embedding.call(idx)
        position_embeddings = @position_embedding.call(idx)

        x = @dropout.call(token_embeddings + position_embeddings)
        @blocks.each { |block| x = block.call(x) }
        x = @ln_f.call(x)
        @lm_head.call(x)
      end

      def generate(input_ids, max_new_tokens:, temperature: 1.0, top_k: nil)
        generated = input_ids.clone
        max_new_tokens.times do
          total_length = generated.shape[1]
          start = [total_length - config[:block_size], 0].max
          length = total_length - start
          idx_cond = generated.narrow(1, start, length)
          logits = forward(idx_cond)
          logits = logits.narrow(1, logits.shape[1] - 1, 1).squeeze(1) / temperature
          logits = top_k_filter(logits, top_k) if top_k
          probs = Torch::NN::Functional.softmax(logits, dim: -1)
          next_token = Torch.multinomial(probs, num_samples: 1)
          generated = Torch.cat([generated, next_token], dim: 1)
        end
        generated
      end

      private

        def top_k_filter(logits, k)
          values, _ = Torch.topk(logits, k)
          min_values = values.narrow(1, values.shape[1] - 1, 1)
          logits.masked_fill(logits < min_values, -Float::INFINITY)
        end
    end
  end
end
