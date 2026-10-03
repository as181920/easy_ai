module EasyAILearning
  module Course
    # All generators use a local RNG. Train/validation/test use independent seeds.
    module Data
      module_function

      def classification(count: 96, seed: 1337)
        rng = Random.new(seed)
        x = Array.new(count) { Array.new(2) { rng.rand * 2 - 1 } }
        y = x.map { |a, b| a * b > 0 ? 1 : 0 }
        [x, y]
      end

      def regression(count: 64, seed: 1337)
        rng = Random.new(seed)
        x = Array.new(count) { [rng.rand * 2 - 1] }
        [x, x.map { |v| [2 * v.first + 0.3] }]
      end

      def manifold(count: 64, seed: 1337)
        rng = Random.new(seed)
        Array.new(count) do
          t = rng.rand * 2 * Math::PI
          [Math.cos(t), Math.sin(t), Math.cos(t) * 0.5, Math.sin(t) * 0.5]
        end
      end

      def images(count: 64, seed: 1337, size: 8)
        rng = Random.new(seed)
        labels = Array.new(count) { |i| i % 2 }.shuffle(random: rng)
        images = labels.map do |label|
          position = rng.rand(1...(size - 1))
          [Array.new(size) do |row|
            Array.new(size) do |column|
              on = label.zero? ? column == position : row == position
              (on ? 0.85 : 0.0) + rng.rand * 0.15
            end
          end]
        end
        [images, labels]
      end

      def sequences(count: 64, seed: 1337, length: 5, vocab: 6)
        rng = Random.new(seed)
        x = Array.new(count) do
          start = rng.rand(vocab)
          Array.new(length) { |i| (start + i) % vocab }
        end
        [x, x.map { |row| row.map { |token| (token + 1) % vocab } }]
      end

      def delayed_copy(count: 64, seed: 1337, length: 10)
        rng = Random.new(seed)
        first = Array.new(count) { rng.rand(6) }
        [first.map { |token| [token] + Array.new(length - 1, 6) }, first]
      end

      def reversal(count: 64, seed: 1337, length: 4, symbols: 4)
        rng = Random.new(seed)
        source = Array.new(count) { Array.new(length) { rng.rand(3...(3 + symbols)) } }
        target = source.map { |row| row.reverse + [2] } # PAD=0, BOS=1, EOS=2
        decoder = target.map { |row| [1] + row[0...-1] }
        [source, decoder, target]
      end

      def mixture(count: 64, seed: 1337)
        rng = Random.new(seed)
        Array.new(count) { |i| [(i.even? ? -1.0 : 1.0) + (rng.rand - 0.5) * 0.3, (rng.rand - 0.5) * 0.3] }
      end

      def tensor(values, device: Torch.device("cpu"), integer: false)
        Torch.tensor(values, dtype: integer ? :int64 : :float32, device: device)
      end
    end
  end
end
