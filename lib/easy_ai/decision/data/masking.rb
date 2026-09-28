module EasyAI
  module Decision
    module Data
      class Masking
        def initialize(tokenizer:, config:)
          @tokenizer, @config = tokenizer, config
        end

        def call(rows, seed:, device: "cpu")
          rng = Random.new(seed)
          length = @config[:input]["state_max_tokens"]
          sequences = rows.map do |row|
            tokens = @tokenizer.encode(row.fetch("text"))
            raise ArgumentError, "MLM text contains no tokens" if tokens.empty?
            # Random windows, rather than always discarding document endings.
            start = tokens.length > length - 2 ? rng.rand(tokens.length - (length - 2) + 1) : 0
            [@tokenizer.id(:cls)] + tokens.slice(start, length - 2) + [@tokenizer.id(:eos)]
          end
          width = sequences.map(&:length).max
          positions, targets, masks = [], [], []
          sequences.each_with_index do |ids, row|
            valid = (1...(ids.length - 1)).to_a
            chosen = valid.select { rng.rand < @config[:training]["mask_probability"] }
            chosen = [valid.sample(random: rng)] if chosen.empty?
            chosen.each do |column|
              positions << row * width + column
              targets << ids[column]
              r = rng.rand
              ids[column] = @tokenizer.id(:mask) if r < 0.8
              ids[column] = rng.rand(Tokenizers::ByteBpe::BYTE_OFFSET...@tokenizer.vocab_size) if r >= 0.8 && r < 0.9
            end
            masks << Array.new(width) { |i| i < ids.length }
            ids.concat(Array.new(width - ids.length, @tokenizer.id(:pad)))
          end
          { ids: Torch.tensor(sequences, dtype: :int64, device: device), mask: Torch.tensor(masks, dtype: :bool, device: device),
           positions: Torch.tensor(positions, dtype: :int64, device: device), targets: Torch.tensor(targets, dtype: :int64, device: device) }
        end
      end
    end
  end
end
