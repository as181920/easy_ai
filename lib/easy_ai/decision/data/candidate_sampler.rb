module EasyAI
  module Decision
    module Data
      # Only for a shared label catalog per language/question, such as intent routing.
      # Arbitrary per-request candidate meanings must keep resampling disabled.
      class CandidateSampler
        def initialize(dataset)
          @catalogs = Hash.new { |hash, key| hash[key] = {} }
          dataset.each do |example|
            catalog = @catalogs[[example.language, example.question]]
            example.options.each do |option|
              id, text = option.values_at("id", "text")
              if catalog.key?(id) && catalog[id] != text
                raise ArgumentError, "Negative resampling requires consistent option descriptions for #{id}"
              end
              catalog[id] = text
            end
          end
        end

        def call(example, rng:)
          catalog = @catalogs.fetch([example.language, example.question])
          ids = catalog.keys - [example.target]
          chosen = ([example.target] + ids.sample(example.options.length - 1, random: rng)).shuffle(random: rng)
          options = chosen.map { |id| { "id" => id, "text" => catalog.fetch(id) } }
          Example.new(example.to_h.merge("options" => options))
        end
      end
    end
  end
end
