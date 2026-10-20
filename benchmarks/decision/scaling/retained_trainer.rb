module DecisionScaling
  class RetainedTrainer < SupervisedTrainer
    RETENTION_SIGNATURE = { "version" => 1, "initial_known_updates" => 500, "maximum_known_updates" => 1000,
      "corrective_known_mix" => [24, 4, 4], "joint" => "unchanged", "unit" => "complete four-query known groups" }.freeze

    # Generic Trainer.resume clears stop reasons. Preserve them here so only
    # the explicitly authorized fork can clear a terminal primary outcome.
    def self.resume(checkpoint, dataset:, output:, fork_from_original: false)
      loaded = EasyAI::Decision::Checkpoint.load(checkpoint)
      new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: output,
        task: loaded.fetch(:metadata).fetch("training").fetch("task").to_sym,
        restored: loaded.fetch(:metadata), fork_from_original: fork_from_original)
    end

    def initialize(**kwargs)
      fork = kwargs.delete(:fork_from_original)
      restored = kwargs[:restored]
      if fork
        state = restored&.fetch("training")
        unless state && state.fetch("step") == 500 && !state.fetch("supervised_known_passed") && state.fetch("supervised_mastery_checks") == 1 && state["stop_reason"]
          raise ArgumentError, "Retention fork requires the measured terminal 500-step primary"
        end
        training = state.deep_dup
        training.delete("stop_reason")
        training["retained_mastery_checks"] = training.fetch("supervised_mastery_checks")
        training["retained_sampling"] = RETENTION_SIGNATURE
        kwargs[:restored] = restored.merge("training" => training)
      elsif restored && restored.dig("training", "retained_sampling") != RETENTION_SIGNATURE
        raise ArgumentError, "Corrective retention sampler changed on resume"
      end
      super(**kwargs)
    end

    def save_checkpoint
      state["retained_sampling"] = RETENTION_SIGNATURE
      super
    end

    private

      def sample_indices(size, rng:)
        return super if state.fetch("step") < 500 || state.fetch("supervised_known_passed", false)
        raise ArgumentError, "Retention effective batch must be 32" unless size == 32
        step = state.fetch("step")
        result = %w[en-US zh-CN].flat_map do |language|
          3.times.flat_map { |index| @known.fetch(["binary", language, %w[Actor Event][(step + index) % 2]]).sample(random: rng) }
        end
        4.times do |index|
          language = %w[en-US zh-CN][index % 2]
          target = %w[no yes][(step + index / 2) % 2]
          result << @natural.fetch([language, 2, target]).sample(random: rng)
        end
        4.times { |index| result << @routing.fetch(%w[en-US zh-CN][index % 2]).sample(random: rng) }
        result
      end
  end
end
