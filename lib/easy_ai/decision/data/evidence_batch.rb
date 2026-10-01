module EasyAI
  module Decision
    module Data
      # Structural newline boundaries only. Gold indices are targets, never input masks.
      class EvidenceBatch
        def initialize(tokenizer)
          @tokenizer, @cache = tokenizer, {}
        end

        def call(examples, states, width, device: "cpu")
          masks = examples.each_with_index.map { |row, index| sentence_masks(row.state, states[index], width) }
          count = masks.map(&:size).max
          masks.map! { |lines| lines + Array.new(count - lines.size) { Array.new(width, false) } }
          { sentence_mask: Torch.tensor(masks, dtype: :bool, device: device),
           evidence_targets: Torch.tensor(examples.map { |row| row.evidence_index || -100 }, dtype: :int64, device: device) }
        end

        private

        def sentence_masks(text, state, width)
          encoded = @cache.delete(text) || @tokenizer.encode_with_offsets(text)
          @cache[text] = encoded
          @cache.shift while @cache.size > 1024
          ids, offsets = encoded
          raise ArgumentError, "Evidence alignment cannot use truncated state tokens" unless state[1...-1] == ids
          cursor = 0
          text.split("\n", -1).map do |line|
            start, stop = cursor, cursor + line.bytesize
            cursor = stop + 1
            content = offsets.map { |left, right| left < stop && right > start }
            [false, *content, false] + Array.new(width - state.size, false)
          end
        end
      end
    end
  end
end
