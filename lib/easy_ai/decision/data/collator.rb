module EasyAI
  module Decision
    module Data
      class Collator
        attr_reader :tokenizer, :config, :truncated

        def initialize(tokenizer:, config:)
          @tokenizer, @config, @truncated = tokenizer, config, 0
          @token_cache = {}
        end

        def call(examples, device: "cpu")
          states = examples.map { |example| state_tokens(example.state) }
          candidates = examples.map { |example| example.options.map { |option| option_tokens(example.question, option.fetch("text")) } }
          max_k = candidates.map(&:length).max
          max_m = candidates.flatten(1).map(&:length).max
          max_l = states.map(&:length).max
          option_ids, option_masks, candidate_masks, answer_masks = [], [], [], []
          candidates.each do |options|
            candidate_masks << Array.new(max_k) { |i| i < options.length }
            # Dummy candidates have CLS: no all-masked attention rows / NaN gradients.
            padded = options + Array.new(max_k - options.length) { [tokenizer.id(:cls)] }
            option_ids << padded.map { |ids| pad(ids, max_m) }
            option_masks << padded.map { |ids| Array.new(max_m) { |i| i < ids.length } }
            answer_masks << padded.map do |ids|
              separator = ids.index(tokenizer.id(:sep)) || ids.length
              Array.new(max_m) { |i| i > separator && i < ids.length - 1 }
            end
          end
          batch = {
            state_ids: tensor(states.map { |ids| pad(ids, max_l) }, :int64, device),
            state_mask: tensor(states.map { |ids| Array.new(max_l) { |i| i < ids.length } }, :bool, device),
            option_ids: tensor(option_ids, :int64, device), option_mask: tensor(option_masks, :bool, device),
            answer_mask: tensor(answer_masks, :bool, device),
            candidate_mask: tensor(candidate_masks, :bool, device),
            targets: tensor(examples.map { |e| e.target_index || 0 }, :int64, device)
          }
          batch.merge!(joint_batch(states, candidates, max_k, device)) if config[:model]["encoding_mode"] == "joint"
          batch
        end

        def state_tokens(text)
          tokens = fit(encode(text), config[:input]["state_max_tokens"] - 2, "state")
          [tokenizer.id(:cls)] + tokens + [tokenizer.id(:eos)]
        end

        def option_tokens(question, text)
          q, a = encode(question), encode(text)
          budget = config[:input]["question_option_max_tokens"] - 3
          if q.length + a.length > budget
            raise ArgumentError, "question+option exceeds #{budget} content tokens" if config[:input]["truncation"] == "error"
            @truncated += 1
            # Preserve content from both the question and option.
            q_budget = [q.length, [budget / 2, budget - a.length].max].min
            q, a = q.first(q_budget), a.first(budget - q_budget)
          end
          [tokenizer.id(:cls)] + q + [tokenizer.id(:sep)] + a + [tokenizer.id(:eos)]
        end

        private

        def joint_batch(states, candidates, max_k, device)
          sequences, answers = [], []
          states.each_with_index do |state, i|
            rows = candidates[i] + Array.new(max_k - candidates[i].length) { [tokenizer.id(:cls)] }
            sequences << rows.map { |option| state[0...-1] + [tokenizer.id(:sep)] + option.drop(1) }
            answers << rows.map do |option|
              separator = option.index(tokenizer.id(:sep)) || option.length
              Array.new(state.length, false) + (1...option.length).map { |j| j > separator && j < option.length - 1 }
            end
          end
          width = sequences.flatten(1).map(&:length).max
          { joint_ids: tensor(sequences.map { |rows| rows.map { |ids| pad(ids, width) } }, :int64, device),
            joint_mask: tensor(sequences.map { |rows| rows.map { |ids| Array.new(width) { |j| j < ids.length } } }, :bool, device),
            joint_answer_mask: tensor(answers.map { |rows| rows.map { |mask| mask + Array.new(width - mask.length, false) } }, :bool, device) }
        end

        def encode(text)
          tokens = @token_cache.delete(text) || tokenizer.encode(text).freeze
          @token_cache[text] = tokens if tokens.length <= 4096
          @token_cache.shift while @token_cache.length > 1024
          tokens
        end

        def fit(tokens, limit, field)
          return tokens if tokens.length <= limit
          raise ArgumentError, "#{field} exceeds #{limit} content tokens" if config[:input]["truncation"] == "error"
          @truncated += 1
          tokens.first(limit)
        end

        def pad(ids, size)
          ids + Array.new(size - ids.length, tokenizer.id(:pad))
        end

        def tensor(values, dtype, device)
          Torch.tensor(values, dtype: dtype, device: device)
        end
      end
    end
  end
end
