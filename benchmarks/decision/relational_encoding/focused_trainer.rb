module DecisionRelational
  # A bounded curriculum intervention, not an inference rule or new label source.
  class FocusedTrainer < EasyAI::Decision::JudgmentTrainer
    SIGNATURE = { "version" => 1, "controlled_only_known_until" => 2000, "unit" => "four-query binding/assertion group" }.freeze

    def initialize(**kwargs)
      if kwargs[:restored] && kwargs[:restored].dig("training", "focused_sampling") != SIGNATURE
        raise ArgumentError, "Focused curriculum resume mismatch"
      end
      units = Hash.new { |hash, key| hash[key] = [] }
      kwargs.fetch(:dataset).each_with_index do |row, index|
        next unless row.source.start_with?("Factual-") && !row.source.end_with?("Single") && row.options.size == 2
        units[[row.language, row.group_id, row.state]] << [index, row]
      end
      @focused_units = Hash.new { |hash, key| hash[key] = [] }
      units.each do |(language, _, _), entries|
        unless entries.size == 4 && entries.map { |_, row| row.target }.sort == %w[no no yes yes] && entries.map { |_, row| row.state }.uniq.size == 1
          raise ArgumentError, "Focused sampling requires complete known binding/assertion groups"
        end
        @focused_units[language] << entries.map(&:first)
      end
      raise ArgumentError, "Focused sampling needs both languages" unless %w[en-US zh-CN].all? { |language| @focused_units[language].any? }
      super
    end

    def save_checkpoint
      state["focused_sampling"] = SIGNATURE
      super
    end

    private

      def sample_indices(size, rng:)
        return super unless state.fetch("judgment_stage") == "known" && state.fetch("step") < SIGNATURE.fetch("controlled_only_known_until")
        raise ArgumentError, "Focused effective batch must be 32" unless size == 32
        Array.new(size / 4) { |index| @focused_units.fetch(%w[en-US zh-CN][index % 2]).sample(random: rng) }.flatten
      end
  end

  module_function

  def focused_variant?(variant)
    %w[focused focused-candidate].include?(variant.to_s)
  end

  def pilot_directory(root, variant)
    raise ArgumentError, "Unknown pilot variant" unless %w[standard focused focused-candidate].include?(variant.to_s)
    File.join(root, variant.to_s == "standard" ? "pilot" : "pilot-#{variant}")
  end

  def authorize_focused!(root)
    authorize_pilot_failure!(root, "standard")
  end

  def authorize_pilot_failure!(root, variant)
    directory = pilot_directory(root, variant); path = File.join(directory, "seed-1337/summary.json")
    raise "Standard pilot incomplete" unless File.exist?(path)
    primary = JSON.parse(File.read(path))
    protocol = JSON.parse(File.read(File.join(directory, "protocol.json")))
    terminal = primary.fetch("step") == protocol.dig("config", "training", "steps") ||
      (primary.fetch("step") == protocol.fetch("known_budget", 1000) && !primary.fetch("known_mastery_passed"))
    raise "Standard pilot is not terminal" unless terminal
    return if primary.fetch("selected").empty?
    confirmation = File.join(directory, "seed-2027/summary.json")
    raise "Confirm eligible standard profiles before an alternative" unless File.exist?(confirmation)
    secondary = JSON.parse(File.read(confirmation))
    raise "Standard recipe has a confirmed eligible profile" unless (primary.fetch("selected").keys & secondary.fetch("selected").keys).empty?
  end
end
