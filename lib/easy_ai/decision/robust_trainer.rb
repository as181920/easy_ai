module EasyAI
  module Decision
    class RobustTrainer < CandidateTrainer
      def initialize(**kwargs)
        previous = kwargs[:restored]&.dig("training", "robust_sampling")
        raise ArgumentError, "Resume robust sampling mismatch" if kwargs[:restored] && previous != Data::RobustSampler::SIGNATURE
        cfg = kwargs.fetch(:model).config
        unless cfg[:training].fetch("choice_microbatch") * cfg[:training].fetch("gradient_accumulation") == 32 &&
            %w[balance_sources balance_labels paired_sampling resample_negatives].none? { |key| cfg[:training].fetch(key) } &&
            kwargs.fetch(:task, :choice).to_sym == :choice && !kwargs[:distillation] && !cfg[:model]["evidence_head"]
          raise ArgumentError, "Robust trainer requires a fixed 32-row CE choice batch without auxiliary objectives"
        end
        @robust_sampler = Data::RobustSampler.new(kwargs.fetch(:dataset))
        super
      end

      def save_checkpoint
        state["robust_sampling"] = Data::RobustSampler::SIGNATURE
        super
      end

      private

      def sample_indices(size, rng:)
        @robust_sampler.sample(size, rng: rng, step: state.fetch("step"))
      end
    end
  end
end
