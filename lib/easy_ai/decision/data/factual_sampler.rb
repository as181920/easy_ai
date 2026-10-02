module EasyAI
  module Decision
    module Data
      # Same row schedule for both objectives, independent of model scores.
      class FactualSampler
        def initialize(dataset)
          @buckets = Hash.new { |hash, key| hash[key] = Hash.new { |languages, language| languages[language] = [] } }
          pairs = Hash.new { |hash, key| hash[key] = [] }
          dataset.each_with_index do |row, index|
            if row.source == "Factual-Contrast"
              key = row.contrast_groups.fetch("fact_flip") { raise ArgumentError, "Missing factual pair" }
              pairs[[row.language, key]] << [index, row]
            else
              kind = row.source == "MASSIVE-Scenario" ? "routing" : "natural"
              @buckets[kind][row.language] << index
            end
          end
          pairs.each_value do |rows|
            examples = rows.map(&:last)
            validate_pair!(examples)
            @buckets["contrast"][examples.first.language] << rows.map(&:first)
          end
          unless %w[natural contrast routing].all? { |kind| @buckets[kind].keys.sort == %w[en-US zh-CN] }
            raise ArgumentError, "Factual mixture requires all three buckets in both languages"
          end
        end

        def sample(size, rng:, step:)
          raise ArgumentError, "Factual effective batch must be 32" unless size == 32
          # Across five updates: exactly 50% natural, 30% contrasts, 20% replay.
          contrast = step % 5 == 4 ? 8 : 10
          natural = Array.new(16) { draw("natural", rng) }
          paired = Array.new(contrast / 2) { draw("contrast", rng) }.flatten
          routing = Array.new(16 - contrast) { draw("routing", rng) }
          natural + paired + routing
        end

        private

        def draw(kind, rng)
          languages = @buckets.fetch(kind)
          languages.fetch(languages.keys.sort.sample(random: rng)).sample(random: rng)
        end

        def validate_pair!(rows)
          unless rows.size == 2 && rows.map(&:target).sort == %w[no yes] && rows.map(&:group_id).uniq.size == 1 &&
            rows.map(&:question).uniq.size == 1 && rows.map { |row| row.options.sort_by { |option| option.fetch("id") } }.uniq.size == 1 &&
            rows.map(&:state).uniq.size == 2
            raise ArgumentError, "Factual pairs require distinct states, identical candidates/questions and opposite gold labels in one group"
          end
        end
      end
    end
  end
end
