module DecisionScaling
  class SupervisedTrainer < EasyAI::Decision::CandidateTrainer
    SIGNATURE = { "version" => 1, "known_updates" => 500, "effective_batch" => 32,
      "known_mix" => [16, 8, 8], "joint_mix" => [12, 8, 6, 6],
      "balance" => "language/axis controlled units; target-balanced natural; language-balanced routing",
      "unknown_requires_retained_mastery" => true }.freeze

    def initialize(**kwargs)
      if kwargs[:restored] && kwargs[:restored].dig("training", "scale_supervised_sampling") != SIGNATURE
        raise ArgumentError, "Supervised scaling sampler changed on resume"
      end
      @known = Hash.new { |hash, key| hash[key] = [] }
      @unknown = Hash.new { |hash, key| hash[key] = [] }
      @natural = Hash.new { |hash, key| hash[key] = [] }
      @routing = Hash.new { |hash, key| hash[key] = [] }
      units = Hash.new { |hash, key| hash[key] = [] }
      pairs = Hash.new { |hash, key| hash[key] = [] }
      kwargs.fetch(:dataset).each_with_index do |row, index|
        if row.source.start_with?("Factual-V06-")
          next if row.source.end_with?("Single")
          axis = row.source.split("-").last
          profile = row.options.size == 2 ? "binary" : "joint"
          if row.target == "unknown"
            pairs[[row.language, row.contrast_groups.fetch("question_flip")]] << [index, row]
          else
            units[[profile, row.language, axis, row.group_id, row.state]] << [index, row]
          end
        elsif row.source == "MASSIVE-Scenario"
          raise ArgumentError, "Routing replay requires all eighteen candidates" unless row.options.size == 18
          @routing[row.language] << index
        else
          unless %w[BoolQ OCNLI].include?(row.source) && [2, 3].include?(row.options.size) && %w[yes no unknown].include?(row.target) && (row.target != "unknown" || row.options.size == 3)
            raise ArgumentError, "Unsupported reviewed natural row"
          end
          @natural[[row.language, row.options.size, row.target]] << index
        end
      end
      units.each do |(profile, language, axis, _, _), entries|
        raise ArgumentError, "Incomplete supervised known unit" unless entries.size == 4 && entries.map { |_, row| row.target }.sort == %w[no no yes yes]
        @known[[profile, language, axis]] << entries.map(&:first)
      end
      pairs.each do |(language, _), entries|
        unless entries.size == 2 && entries.all? { |_, row| row.target == "unknown" && row.options.size == 3 } &&
            entries.map { |_, row| row.state }.uniq.size == 1 && entries.map { |_, row| row.question }.uniq.size == 2
          raise ArgumentError, "Incomplete supervised unknown pair"
        end
        @unknown[language] << entries.map(&:first)
      end
      %w[en-US zh-CN].each do |language|
        %w[binary joint].product(%w[Actor Event]).each do |profile, axis|
          raise ArgumentError, "Missing supervised known bucket" if @known[[profile, language, axis]].empty?
        end
        raise ArgumentError, "Missing supervised unknown/routing bucket" if @unknown[language].empty? || @routing[language].empty?
        %w[yes no].each { |target| raise ArgumentError, "Missing natural binary target" if @natural[[language, 2, target]].empty? }
      end
      super
    end

    def save_checkpoint
      state["scale_supervised_sampling"] = SIGNATURE
      super
    end

    private

      def sample_indices(size, rng:)
        raise ArgumentError, "Supervised scaling batch must be 32" unless size == 32
        step = state.fetch("step")
        joint = step >= SIGNATURE.fetch("known_updates") && state.fetch("supervised_known_passed", false)
        profile = joint ? "joint" : "binary"
        buckets = %w[en-US zh-CN].product(%w[Actor Event])
        result = Array.new(joint ? 3 : 4) do |index|
          language, axis = buckets.fetch((step + index) % buckets.size)
          @known.fetch([profile, language, axis]).sample(random: rng)
        end.flatten
        if joint
          4.times { |index| result.concat(@unknown.fetch(%w[en-US zh-CN][(step + index) % 2]).sample(random: rng)) }
        end
        (joint ? 6 : 8).times do |index|
          language = %w[en-US zh-CN][index % 2]
          keys = @natural.keys.select { |lang, count, _| lang == language && (joint || count == 2) }
          # Choose targets before examples: the fresh Chinese source pool is
          # class-skewed, so uniform example sampling would favor entailment.
          targets = keys.map(&:last).uniq.sort
          target = targets.fetch((step + index / 2) % targets.size)
          key = keys.select { |_, _, label| label == target }.sample(random: rng)
          result << @natural.fetch(key).sample(random: rng)
        end
        (joint ? 6 : 8).times { |index| result << @routing.fetch(%w[en-US zh-CN][index % 2]).sample(random: rng) }
        result
      end
  end
end
