module EasyAI
  module Decision
    module Data
      class JudgmentSampler
        SIGNATURE = { "version" => 4, "controlled" => 0.6, "natural" => 0.2, "routing" => 0.2, "cycle" => 5 }.freeze

        def initialize(dataset)
          @pairs = Hash.new { |hash, key| hash[key] = [] }
          @other = Hash.new { |hash, key| hash[key] = [] }
          pairs = Hash.new { |hash, key| hash[key] = [] }
          dataset.each_with_index do |row, index|
            if row.source.start_with?("Factual-")
              profile = row.options.size == 3 ? "joint" : row.source.end_with?("Single") ? "single" : "binary"
              pairs[[profile, row.language, row.target == "unknown" ? "unknown" : "known", row.contrast_groups.fetch("question_flip")]] << [index, row]
            else
              kind = row.source == "MASSIVE-Scenario" ? "routing" : "natural"
              @other[[kind, row.language, kind == "routing" ? 0 : row.options.size]] << index
            end
          end
          pairs.each do |(profile, language, kind, _), entries|
            rows = entries.map(&:last)
            targets = kind == "known" ? %w[no yes] : %w[unknown unknown]
            unless rows.size == 2 && rows.map(&:target).sort == targets && rows.map(&:state).uniq.size == 1 && rows.map(&:question).uniq.size == 2 && rows.map(&:options).uniq.size == 1
              raise ArgumentError, "Judgment sampling needs coherent complete question pairs"
            end
            @pairs[[profile, language, kind]] << entries.map(&:first)
          end
          %w[en-US zh-CN].each do |language|
            %w[single binary joint].each { |profile| raise ArgumentError, "Missing known bucket" if @pairs[[profile, language, "known"]].empty? }
            raise ArgumentError, "Missing unknown bucket" if @pairs[["joint", language, "unknown"]].empty?
            %w[natural routing].each { |kind| raise ArgumentError, "Missing #{kind} bucket" if @other[[kind, language, kind == "routing" ? 0 : 2]].empty? }
          end
        end

        def sample(size, rng:, step:, stage:)
          raise ArgumentError, "Judgment effective batch must be 32" unless size == 32
          raise ArgumentError, "Unknown curriculum stage" unless %w[known joint].include?(stage)
          core = step % 5 == 4 ? 24 : 18
          profile = stage == "joint" ? "joint" : step < 200 ? "single" : "binary"
          unit = stage == "joint" ? 6 : 2
          result = Array.new(core / unit) do |index|
            language = %w[en-US zh-CN][(step * (18 / unit) + index) % 2]
            if stage == "joint"
              Array.new(2) { @pairs[[profile, language, "known"]].sample(random: rng) }.flatten + @pairs[[profile, language, "unknown"]].sample(random: rng)
            else
              @pairs[[profile, language, "known"]].sample(random: rng)
            end
          end.flatten
          %w[natural routing].each do |kind|
            count = (32 - core) / 2
            count.times do |index|
              language = %w[en-US zh-CN][(step * 7 + index) % 2]
              result << @other[[kind, language, kind == "routing" ? 0 : 2]].sample(random: rng)
            end
          end
          result
        end
      end
    end
  end
end
