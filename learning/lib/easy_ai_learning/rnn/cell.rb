module EasyAILearning
  module RNN
    class Cell < Torch::NN::Module
      attr_reader :input_projection, :state_projection, :kind, :hidden_size

      def initialize(input:, hidden: 8, kind: :rnn)
        super()
        kind = kind.to_s.to_sym
        raise ArgumentError, "Unknown recurrent cell" unless %i[rnn lstm gru].include?(kind)
        @kind, @hidden_size = kind, hidden
        gates = { rnn: 1, lstm: 4, gru: 3 }.fetch(kind)
        @input_projection = Torch::NN::Linear.new(input, hidden * gates)
        @state_projection = Torch::NN::Linear.new(hidden, hidden * gates)
      end

      def initial(batch, device: Torch.device("cpu"))
        h = Torch.zeros([batch, hidden_size], device: device)
        kind == :lstm ? [h, Torch.zeros_like(h)] : h
      end

      def forward(x, state)
        h = kind == :lstm ? state.first : state
        input, recurrent = input_projection.call(x), state_projection.call(h)
        return Torch.tanh(input + recurrent) if kind == :rnn
        if kind == :lstm
          i, f, g, o = 4.times.map { |j| (input + recurrent).narrow(1, j * hidden_size, hidden_size) }
          cell = Torch.sigmoid(f) * state.last + Torch.sigmoid(i) * Torch.tanh(g)
          return [Torch.sigmoid(o) * Torch.tanh(cell), cell]
        end
        ix, hx = 3.times.map { |j| input.narrow(1, j * hidden_size, hidden_size) }, 3.times.map { |j| recurrent.narrow(1, j * hidden_size, hidden_size) }
        reset, update = Torch.sigmoid(ix[0] + hx[0]), Torch.sigmoid(ix[1] + hx[1])
        candidate = Torch.tanh(ix[2] + reset * hx[2])
        (1 - update) * candidate + update * h
      end
    end
  end
end
