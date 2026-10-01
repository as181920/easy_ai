module EasyAI
  module Decision
    module Data
      # Schema conversion only; reference labels and latent factors never enter inputs.
      class BenchmarkAdapter
        def self.jev(row, tier:)
          raise ArgumentError, "Only measured public tasks are supported" unless row.fetch("split") == "public" &&
            !row["expected"].nil? && !row.fetch("provenance")["exclude_reason"]
          spec = row.fetch("question")
          labels = row.fetch("labels").map(&:to_s)
          descriptions = criteria(spec)
          options = labels.map do |label|
            key = spec.fetch("type") == "noul" ? { "no" => "false", "yes" => "true" }.fetch(label) : label
            { "id" => label, "text" => "#{key}: #{descriptions.fetch(key)}" }
          end
          base(row.fetch("id"), row["group"] || row.fetch("id"), "JevBench/#{tier}",
            row.fetch("state").is_a?(String) ? row.fetch("state") : serialize(row.fetch("state")), spec, options, row.fetch("expected").to_s)
            .merge("benchmark" => { "type" => spec.fetch("type"), "tier" => tier, "family" => row.fetch("family") })
        end

        def self.typed(row)
          raise ArgumentError, "Only test cases are supported" unless row.fetch("split") == "test"
          questions = parse(row.fetch("questions"))
          gold = parse(row.fetch("gold"))
          raise ArgumentError, "Expected five decisions" unless questions.size == 5
          questions.map do |name, spec|
            descriptions = criteria(spec)
            answer = gold.fetch(name)
            probabilities = answer.fetch("probabilities").transform_keys(&:to_s)
            raise ArgumentError, "Reference labels differ" unless probabilities.keys.to_set == descriptions.keys.to_set
            base("#{row.fetch('id')}:#{name}", row.fetch("id"), "TypedDecisions/#{row.fetch('workflow')}", row.fetch("state"),
              spec, descriptions.map { |id, text| { "id" => id, "text" => "#{id}: #{text}" } }, answer.fetch("label").to_s)
              .merge("benchmark" => { "type" => spec.fetch("type"), "workflow" => row.fetch("workflow"), "question_name" => name,
                "reference_probabilities" => probabilities })
          end
        end

        def self.criteria(spec)
          case spec.fetch("type")
          when "score" then spec.fetch("criteria").each_with_index.to_h { |text, index| [index.to_s, text] }
          when "choice" then spec.fetch("criteria").transform_keys(&:to_s)
          when "noul" then spec["criteria"] || { "false" => "false", "true" => "true" }
          else raise ArgumentError, "Unknown primitive"
          end
        end

        def self.base(id, group, source, state, spec, options, target)
          row = { "id" => id, "group_id" => "shared-benchmark:#{group}", "source" => source, "language" => "en-US",
            "state" => state, "question" => spec.fetch("instructions"), "options" => options, "target" => target }
          Example.new(row)
          row
        end

        def self.parse(value)
          value.is_a?(String) ? JSON.parse(value) : value
        end

        # Canonical structured state: deterministic key order and conventional spacing.
        def self.serialize(value)
          case value
          when Hash then "{#{value.keys.sort.map { |key| "#{JSON.generate(key)}: #{serialize(value.fetch(key))}" }.join(', ')}}"
          when Array then "[#{value.map { |item| serialize(item) }.join(', ')}]"
          else JSON.generate(value)
          end
        end
      end
    end
  end
end
