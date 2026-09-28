module EasyAI
  module Decision
    module Data
      # Keep complete contrast groups in each optimizer update. GPU recovery may
      # split them across accumulation microbatches; the loss remains per-row CE.
      class PairSampler
        GROUP_SIZES = { "question_flip" => 2, "binding" => 4 }.freeze

        attr_reader :group_size

        def initialize(dataset, strategy: "question_flip")
          @group_size = GROUP_SIZES.fetch(strategy) { raise ArgumentError, "Unknown contrast strategy: #{strategy}" }
          groups = Hash.new { |hash, key| hash[key] = [] }
          dataset.each_with_index do |example, index|
            key = strategy == "question_flip" ? example.contrast_group : example.contrast_groups[strategy]
            raise ArgumentError, "Missing #{strategy} contrast group" unless key
            groups[[example.language, key]] << [index, example.group_id]
          end
          unless groups.any? && groups.values.all? { |rows| rows.length == group_size && rows.map(&:last).uniq.length == 1 }
            raise ArgumentError, "Every contrast group must contain #{group_size} rows in one split group"
          end
          @languages = groups.group_by { |(language, _key), _rows| language }.transform_values do |pairs|
            pairs.map { |_key, rows| rows.map(&:first) }
          end
        end

        def sample(size, rng:)
          raise ArgumentError, "Sample size must be a positive multiple of #{group_size}" unless size.is_a?(Integer) && size > 0 && (size % group_size).zero?
          Array.new(size / group_size) do
            language = @languages.keys.sort.sample(random: rng)
            @languages.fetch(language).sample(random: rng)
          end.flatten
        end
      end
    end
  end
end
