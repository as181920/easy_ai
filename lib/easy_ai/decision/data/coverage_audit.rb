module EasyAI
  module Decision
    module Data
      # Language-independent exposure accounting using declared dataset metadata.
      class CoverageAudit
        def self.call(dataset, visits: nil, input_tokens: nil)
          visits ||= Array.new(dataset.size, 0)
          raise ArgumentError, "Coverage row count mismatch" unless visits.size == dataset.size
          bins = {}
          dataset.each_with_index do |row, index|
            keys = ["all", "source/#{row.source}", "language/#{row.language}", "label/#{row.source}/#{row.target}"]
            keys.each do |key|
              bin = bins[key] ||= { rows: 0, groups: Set.new, seen_rows: 0, seen_groups: Set.new, occurrences: 0 }
              bin[:rows] += 1
              bin[:groups] << row.group_id
              bin[:occurrences] += visits[index]
              if visits[index] > 0
                bin[:seen_rows] += 1
                bin[:seen_groups] << row.group_id
              end
            end
          end
          { "dataset_sha256" => dataset.fingerprint, "input_tokens" => input_tokens,
            "token_scope" => "Committed updates only; unpadded encoder inputs including special tokens and repeated question text. Excludes validation and retried work.",
            "scope" => "Counts declared source/language/gold-label metadata; does not infer language or semantic categories from text.",
            "strata" => bins.transform_values do |bin|
              { "rows" => bin[:rows], "groups" => bin[:groups].size, "seen_rows" => bin[:seen_rows],
                "seen_groups" => bin[:seen_groups].size, "occurrences" => bin[:occurrences],
                "row_coverage" => bin[:seen_rows].fdiv(bin[:rows]), "group_coverage" => bin[:seen_groups].size.fdiv(bin[:groups].size) }
            end }
        end
      end
    end
  end
end
