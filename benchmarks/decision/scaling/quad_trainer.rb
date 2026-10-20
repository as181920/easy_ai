module DecisionScaling
  class QuadTrainer < EasyAI::Decision::CandidateTrainer
    SIGNATURE = { "version" => 1, "unit" => "four-query known group", "balanced" => "language and actor/event axis", "effective_batch" => 32 }.freeze

    def initialize(**kwargs)
      if kwargs[:restored] && kwargs[:restored].dig("training", "scale_sampling") != SIGNATURE
        raise ArgumentError, "Scaling sampler changed on resume"
      end
      groups = Hash.new { |hash, key| hash[key] = [] }
      kwargs.fetch(:dataset).each_with_index do |row, index|
        axis = row.source.split("-").last
        unless row.options.size == 2 && %w[Actor Event].include?(axis)
          raise ArgumentError, "Controlled scaling requires binary actor/event rows"
        end
        groups[[row.language, axis, row.group_id, row.state]] << [index, row]
      end
      @units = Hash.new { |hash, key| hash[key] = [] }
      groups.each do |(language, axis, _, _), entries|
        raise ArgumentError, "Incomplete four-query scaling unit" unless entries.size == 4 && entries.map { |_, row| row.target }.sort == %w[no no yes yes]
        @units[[language, axis]] << entries.map(&:first)
      end
      raise ArgumentError, "Missing language/axis bucket" unless %w[en-US zh-CN].product(%w[Actor Event]).all? { |key| @units[key].any? }
      super
    end

    def save_checkpoint
      state["scale_sampling"] = SIGNATURE
      super
    end

    private

      def sample_indices(size, rng:)
        raise ArgumentError, "Scaling effective batch must be 32" unless size == 32
        %w[en-US zh-CN].product(%w[Actor Event]).flat_map { |key| 2.times.flat_map { @units.fetch(key).sample(random: rng) } }
      end
  end
end
