module EasyAI
  module Decision
    module Data
      # Keep both members of each explicit contrast pair in the same microbatch.
      class PairSampler
        def initialize(dataset)
          groups = Hash.new { |hash, key| hash[key] = [] }
          dataset.each_with_index do |example, index|
            raise ArgumentError, "paired_sampling requires contrast_group on every example" unless example.contrast_group
            groups[[example.language, example.contrast_group]] << [index, example.group_id]
          end
          unless groups.values.all? { |rows| rows.length == 2 && rows.map(&:last).uniq.length == 1 }
            raise ArgumentError, "Every contrast group must contain two rows in one split group"
          end
          @languages = groups.group_by { |(language, _key), _rows| language }.transform_values do |pairs|
            pairs.map { |_key, rows| rows.map(&:first) }
          end
        end

        def sample(size, rng:)
          raise ArgumentError, "Paired sample size must be positive and even" unless size > 0 && size.even?
          Array.new(size / 2) do
            language = @languages.keys.sort.sample(random: rng)
            @languages.fetch(language).sample(random: rng)
          end.flatten
        end
      end
    end
  end
end
