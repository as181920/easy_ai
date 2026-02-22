module EasyAI
  module Data
    class Batcher
      attr_reader :dataset, :batch_size, :device, :rng, :chunk_shift

      def initialize(dataset:, batch_size:, device: Torch.device("cpu"), seed: 1337, chunk_shift: 1)
        @dataset = dataset
        @batch_size = batch_size
        @device = device
        @rng = Random.new(seed)
        @chunk_shift = chunk_shift
      end

      # Build batches on CPU, let trainer move to GPU as needed.
      # This avoids accumulating stale GPU tensors from previous batches.
      #
      # chunk_shift: Number of positions to shift for target.
      #   1 = standard next-token prediction (input[i] → target[i+1])
      #   N = skip-gram style (input[i] → target[i+N])
      def next_batch
        raise ArgumentError, "Dataset too small for block size + chunk_shift" if max_start_index.negative?

        starts = Array.new(batch_size) { rng.rand(0..max_start_index) }
        block_size = dataset.block_size

        x = Torch.zeros(batch_size, block_size, dtype: :int64)
        y = Torch.zeros(batch_size, block_size, dtype: :int64)

        starts.each_with_index do |start, row|
          input_ids = dataset.token_ids[start, block_size]
          target_ids = dataset.token_ids[(start + chunk_shift), block_size]

          x[row] = Torch.tensor(input_ids, dtype: :int64)
          y[row] = Torch.tensor(target_ids, dtype: :int64)
        end

        [x, y]
      end

      private

        def max_start_index
          dataset.length - dataset.block_size - chunk_shift - 1
        end
    end
  end
end
