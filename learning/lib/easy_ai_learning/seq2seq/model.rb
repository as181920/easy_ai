module EasyAILearning
  module Seq2seq
    class Model < Torch::NN::Module
      attr_reader :encoder, :decoder, :embedding

      def initialize(vocab: 7, hidden: 12, attention: false)
        super()
        @embedding = Torch::NN::Embedding.new(vocab, hidden)
        @encoder = RNN::Cell.new(input: hidden, hidden: hidden)
        @decoder = RNN::Cell.new(input: hidden, hidden: hidden)
        @head = Torch::NN::Linear.new(hidden, vocab)
        @attention = attention ? Attention::MultiHead.new(embed_dim: hidden) : nil
        @fusion = attention ? Torch::NN::Linear.new(hidden * 2, hidden) : nil
      end

      def encode(source)
        encode_memory(source).first
      end

      def encode_memory(source)
        state = encoder.initial(source.shape[0], device: source.device)
        vectors, states = embedding.call(source), []
        source.shape[1].times do |t|
          state = encoder.call(vectors.narrow(1, t, 1).squeeze(1), state)
          states << state
        end
        [state, Torch.stack(states, dim: 1)]
      end

      def decode_step(token, state, memory)
        state = decoder.call(embedding.call(token), state)
        features = state
        if @attention
          context = @attention.call(state.unsqueeze(1), memory: memory).squeeze(1)
          features = Torch.tanh(@fusion.call(Torch.cat([state, context], dim: 1)))
        end
        [state, @head.call(features)]
      end

      def forward(source, decoder_tokens)
        state, memory = encode_memory(source)
        outputs = []
        decoder_tokens.shape[1].times do |t|
          state, logits = decode_step(decoder_tokens.narrow(1, t, 1).squeeze(1), state, memory)
          outputs << logits
        end
        Torch.stack(outputs, dim: 1)
      end

      def generate(source, length: 5, bos: 1, eos: 2)
        Torch.no_grad do
          state, memory = encode_memory(source)
          token = Torch.full([source.shape[0]], bos, dtype: :int64, device: source.device)
          done = Torch.zeros([source.shape[0]], dtype: :bool, device: source.device)
          outputs = []
          length.times do
            state, logits = decode_step(token, state, memory)
            predicted = logits.argmax(-1)
            token = predicted.masked_fill(done, eos)
            done = done.logical_or(token.eq(eos))
            outputs << token
          end
          Torch.stack(outputs, dim: 1)
        end
      end
    end
  end
end
