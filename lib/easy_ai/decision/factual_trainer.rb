module EasyAI
  module Decision
    class FactualTrainer < CandidateTrainer
      def initialize(pair_margin_weight: 0.0, pair_margin: 1.0, **kwargs)
        unless pair_margin_weight.is_a?(Numeric) && pair_margin_weight.finite? && pair_margin_weight >= 0 &&
            pair_margin.is_a?(Numeric) && pair_margin.finite? && pair_margin > 0
          raise ArgumentError, "Invalid factual margin configuration"
        end
        @factual_signature = { "version" => 1, "pair_margin_weight" => pair_margin_weight, "pair_margin" => pair_margin }
        previous = kwargs[:restored]&.dig("training", "factual_objective")
        raise ArgumentError, "Resume factual objective mismatch" if kwargs[:restored] && previous != @factual_signature
        cfg = kwargs.fetch(:model).config[:training]
        unless cfg.fetch("choice_microbatch") * cfg.fetch("gradient_accumulation") == 32 &&
            %w[balance_sources balance_labels paired_sampling resample_negatives].none? { |key| cfg.fetch(key) } &&
            kwargs.fetch(:task, :choice).to_sym == :choice && !kwargs[:distillation] && !kwargs.fetch(:model).config[:model]["evidence_head"]
          raise ArgumentError, "Factual trainer needs a fixed 32-row choice batch without other sampling or auxiliary objectives"
        end
        @factual_sampler = Data::FactualSampler.new(kwargs.fetch(:dataset))
        @margin_weight, @margin = pair_margin_weight, pair_margin
        super(**kwargs)
      end

      def save_checkpoint
        state["factual_objective"] = @factual_signature
        super
      end

      private

      def sample_indices(size, rng:)
        @factual_sampler.sample(size, rng: rng, step: state.fetch("step"))
      end

      def loss_for(examples, seed:, teacher: false)
        rng = Random.new(seed ^ 0xF177)
        examples = examples.map { |row| Data::Example.new(row.to_h.merge("options" => row.options.shuffle(random: rng))) }
        @grouped_input_tokens = 0
        @batch_losses = { "choice_loss" => 0.0, "pair_margin_loss" => 0.0 }
        losses = examples.group_by { |row| row.options.size }.values.map do |rows|
          batch = @collator.call(rows, device: device)
          @grouped_input_tokens += @collator.input_token_counts.sum
          logits = model.call(batch)
          ce = Torch::NN::Functional.cross_entropy(logits, batch.fetch(:targets))
          auxiliary = teacher ? PairMarginLoss.call(logits, rows, margin: @margin) : logits.sum * 0
          weight = rows.size.fdiv(examples.size)
          @batch_losses["choice_loss"] += ce.item * weight
          @batch_losses["pair_margin_loss"] += auxiliary.item * weight
          (ce + auxiliary * @margin_weight) * weight
        end
        Torch.stack(losses).sum
      end
    end
  end
end
