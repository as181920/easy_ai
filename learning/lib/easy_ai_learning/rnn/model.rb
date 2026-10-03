module EasyAILearning
  module RNN
    class Model < Torch::NN::Module
      attr_reader :cell, :embedding

      def initialize(vocab: 6, hidden: 8, kind: :rnn)
        super()
        @embedding = Torch::NN::Embedding.new(vocab, hidden)
        @cell = Cell.new(input: hidden, hidden: hidden, kind: kind)
        @head = Torch::NN::Linear.new(hidden, vocab)
      end

      def forward(tokens, state: nil, lengths: nil, truncate: nil)
        raise ArgumentError, "Positive truncation interval required" if truncate && truncate <= 0
        if lengths && (lengths.shape != [tokens.shape[0]] || lengths.min.item < 0 || lengths.max.item > tokens.shape[1])
          raise ArgumentError, "Invalid sequence lengths"
        end
        state ||= cell.initial(tokens.shape[0], device: tokens.device)
        vectors, outputs = embedding.call(tokens), []
        tokens.shape[1].times do |step|
          proposed = cell.call(vectors.narrow(1, step, 1).squeeze(1), state)
          if lengths
            visible = lengths.gt(step).to(dtype: vectors.dtype).unsqueeze(1)
            proposed = blend(proposed, state, visible)
          end
          state = proposed
          h = state.is_a?(Array) ? state.first : state
          outputs << @head.call(h)
          state = state.is_a?(Array) ? state.map(&:detach) : state.detach if truncate && (step + 1) % truncate == 0
        end
        Torch.stack(outputs, dim: 1)
      end

      private

      def blend(proposed, previous, mask)
        return proposed.zip(previous).map { |a, b| a * mask + b * (1 - mask) } if proposed.is_a?(Array)
        proposed * mask + previous * (1 - mask)
      end
    end
  end
end
