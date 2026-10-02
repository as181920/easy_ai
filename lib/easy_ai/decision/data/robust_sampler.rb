module EasyAI
  module Decision
    module Data
      # CE supervision with explicit unknown exposure and complete same-state query pairs.
      class RobustSampler
        SIGNATURE = { "version" => 3, "natural" => 0.3, "known" => 0.3, "unknown" => 0.2, "routing" => 0.2 }.freeze

        def initialize(dataset)
          @buckets = Hash.new { |hash, kind| hash[kind] = Hash.new { |languages, language| languages[language] = [] } }
          pairs = Hash.new { |hash, key| hash[key] = [] }
          dataset.each_with_index do |row, index|
            if row.source.start_with?("Factual-V03-")
              axis = row.contrast_groups.key?("question_flip") ? "question_flip" : "fact_flip"
              key = row.contrast_groups.fetch(axis)
              kind = row.target == "unknown" ? "unknown" : "known"
              pairs[[kind, row.language, axis, key]] << [index, row]
            else
              kind = row.source == "MASSIVE-Scenario" ? "routing" : "natural"
              @buckets[kind][row.language] << index
            end
          end
          pairs.each do |(kind, language, axis, _), items|
            rows = items.map(&:last)
            expected = kind == "known" ? %w[no yes] : %w[unknown unknown]
            coherent = axis == "question_flip" ? rows.map(&:state).uniq.size == 1 && rows.map(&:question).uniq.size == 2 :
              rows.map(&:state).uniq.size == 2 && rows.map(&:question).uniq.size == 1
            unless rows.size == 2 && rows.map(&:target).sort == expected && coherent &&
                rows.map(&:group_id).uniq.size == 1 && rows.map(&:options).uniq.size == 1
              raise ArgumentError, "Robust sampling requires valid complete same-state question pairs"
            end
            @buckets[kind][language] << items.map(&:first)
          end
          unless %w[natural known unknown routing].all? { |kind| @buckets[kind].keys.sort == %w[en-US zh-CN] }
            raise ArgumentError, "Robust mixture requires all four buckets in both languages"
          end
        end

        def sample(size, rng:, step:)
          raise ArgumentError, "Robust effective batch must be 32" unless size == 32
          # Five updates: 48 natural, 48 known, 32 unknown and 32 routing decisions.
          known = step % 5 == 4 ? 8 : 10
          unknown = step % 5 == 4 ? 8 : 6
          natural = step % 5 == 4 ? 8 : 10
          routing = 32 - known - unknown - natural
          Array.new(natural) { draw("natural", rng) } +
            Array.new(known / 2) { draw("known", rng) }.flatten +
            Array.new(unknown / 2) { draw("unknown", rng) }.flatten +
            Array.new(routing) { draw("routing", rng) }
        end

        private

        def draw(kind, rng)
          language = %w[en-US zh-CN].sample(random: rng)
          @buckets.fetch(kind).fetch(language).sample(random: rng)
        end
      end
    end
  end
end
