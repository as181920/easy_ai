module EasyAILearning
  module Transformer
    class EncoderDecoder < Torch::NN::Module
      def initialize(vocab: 7, width: 8, max_length: 8)
        super()
        @embedding = Torch::NN::Embedding.new(vocab, width)
        @position = PositionalEmbeddings.new(block_size: max_length, embedding_dim: width)
        @encoder = EncoderBlock.new(width: width)
        @decoder = Block.new(embed_dim: width, num_heads: 2, dropout: 0)
        @cross = Attention::MultiHead.new(embed_dim: width, num_heads: 2)
        @norm = Torch::NN::LayerNorm.new(width)
        @head = Torch::NN::Linear.new(width, vocab)
      end

      def forward(source, decoder_tokens)
        memory = @encoder.call(@embedding.call(source) + @position.call(source))
        decoded = @decoder.call(@embedding.call(decoder_tokens) + @position.call(decoder_tokens))
        @head.call(@norm.call(decoded + @cross.call(decoded, memory: memory)))
      end

      def generate(source, length: 5, bos: 1)
        Torch.no_grad do
          tokens = Torch.full([source.shape[0], 1], bos, dtype: :int64, device: source.device)
          length.times do
            logits = forward(source, tokens)
            next_token = logits.narrow(1, logits.shape[1] - 1, 1).argmax(-1)
            tokens = Torch.cat([tokens, next_token], dim: 1)
          end
          tokens.narrow(1, 1, length)
        end
      end
    end
  end
end
