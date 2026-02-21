module EasyAI
  module Data
    class Batcher
      attr_reader :dataset, :batch_size, :device, :rng

      def initialize(dataset:, batch_size:, device: Torch.device("cpu"), seed: 1337)
        @dataset = dataset
        @batch_size = batch_size
        @device = device
        @rng = Random.new(seed)
      end

      def next_batch
        raise ArgumentError, "Dataset too small for block size" if max_start_index.negative?

        starts = Array.new(batch_size) { rng.rand(0..max_start_index) }
        block_size = dataset.block_size

        x = Torch.zeros(batch_size, block_size, dtype: :int64, device: device)
        y = Torch.zeros(batch_size, block_size, dtype: :int64, device: device)

        starts.each_with_index do |start, row|
          chunk = dataset.token_ids.slice(start, block_size + 1)
          input_ids = chunk[0...block_size]
          target_ids = chunk[1, block_size]

          x[row] = Torch.tensor(input_ids, dtype: :int64, device: device)
          y[row] = Torch.tensor(target_ids, dtype: :int64, device: device)
        end

        [x, y]
      end

      private

        def max_start_index
          dataset.length - dataset.block_size - 1
        end
    end
  end
end
