module EasyAI
  module Decision
    class JudgmentTrainer < CandidateTrainer
      def initialize(**kwargs)
        previous = kwargs[:restored]&.dig("training", "judgment_sampling")
        raise ArgumentError, "Resume judgment sampling mismatch" if kwargs[:restored] && previous != Data::JudgmentSampler::SIGNATURE
        cfg = kwargs.fetch(:model).config
        unless cfg[:training].fetch("choice_microbatch") * cfg[:training].fetch("gradient_accumulation") == 32 &&
            %w[balance_sources balance_labels paired_sampling resample_negatives].none? { |key| cfg[:training].fetch(key) } &&
            kwargs.fetch(:task, :choice).to_sym == :choice && !kwargs[:distillation] && !cfg[:model]["evidence_head"]
          raise ArgumentError, "Judgment trainer requires fixed 32-row plain CE choice batches"
        end
        @judgment_sampler = Data::JudgmentSampler.new(kwargs.fetch(:dataset))
        super
      end

      def save_checkpoint
        state["judgment_sampling"] = Data::JudgmentSampler::SIGNATURE
        state["judgment_stage"] ||= "known"
        state["mastery_checks"] ||= 0
        super
      end

      def advance!(mastered:)
        raise ArgumentError, "Cannot advance without measured mastery" unless mastered
        raise ArgumentError, "Already in joint stage" unless state.fetch("judgment_stage") == "known"
        state["judgment_stage"] = "joint"
        state["joint_started_at"] = state.fetch("step")
        save_checkpoint
      end

      private

      def sample_indices(size, rng:)
        @judgment_sampler.sample(size, rng: rng, step: state.fetch("step"), stage: state.fetch("judgment_stage"))
      end
    end
  end
end
