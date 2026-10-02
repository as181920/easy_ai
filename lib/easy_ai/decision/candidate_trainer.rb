module EasyAI
  module Decision
    # Full candidate sets, random order and equal weight per example, grouped to avoid padding.
    class CandidateTrainer < Trainer
      private

      def loss_for(examples, seed:, teacher: false)
        rng = Random.new(seed ^ 0xF177)
        examples = examples.map { |row| Data::Example.new(row.to_h.merge("options" => row.options.shuffle(random: rng))) }
        @grouped_input_tokens = 0
        components = Hash.new(0.0)
        losses = examples.group_by { |row| row.options.size }.values.map do |rows|
          weight = rows.size.fdiv(examples.size)
          loss = super(rows, seed: seed, teacher: teacher) * weight
          @grouped_input_tokens += @collator.input_token_counts.sum
          @batch_losses.each { |key, value| components[key] += value * weight } if teacher
          loss
        end
        @batch_losses = components if teacher
        Torch.stack(losses).sum
      end

      def input_token_count
        @grouped_input_tokens
      end
    end
  end
end
