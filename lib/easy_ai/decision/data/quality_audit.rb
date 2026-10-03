require "digest"
require "set"

module EasyAI
  module Decision
    module Data
      # Integrity and reviewer bookkeeping, not semantic inference rules.
      class QualityAudit
        def self.export(rows, per_source: 50)
          buckets = rows.group_by { |row| row.fetch("source").start_with?("Factual-") ? "controlled" : row.fetch("source") }
          selected = buckets.reject { |source, _| source == "MASSIVE-Scenario" }.flat_map do |source, items|
            strata = items.group_by { |row| [row.fetch("language"), row.fetch("target")] }.values
            queues = strata.map { |group| group.sort_by { |row| digest("review:#{row.fetch('id')}") } }
            Array.new(per_source) { |index| queues[index % queues.size].shift }.compact.map do |row|
              row.merge("review_id" => digest("#{source}:#{row.fetch('id')}")[0, 16])
            end
          end.sort_by { |row| digest("blind:#{row.fetch('review_id')}") }
          blind = selected.map do |row|
            row.slice("review_id", "language", "state", "question", "options").merge(
              "contract" => row.fetch("source").start_with?("Factual-") ? "explicit record" : row.fetch("source"))
          end
          [blind, selected]
        end

        def self.adjudicate(selected, reviews)
          raise ArgumentError, "Duplicate review IDs" unless reviews.map { |row| row.fetch("review_id") }.uniq.size == reviews.size
          expected = selected.map { |row| row.fetch("review_id") }.sort
          raise ArgumentError, "Review must cover the exported sample exactly" unless reviews.map { |row| row.fetch("review_id") }.sort == expected
          lookup = reviews.to_h { |row| [row.fetch("review_id"), row] }
          decisions = selected.map do |row|
            review = lookup.fetch(row.fetch("review_id"))
            target = review.fetch("target").to_s
            unless (target == "exclude" || row.fetch("options").any? { |option| option.fetch("id").to_s == target }) &&
                review.fetch("reason").is_a?(String) && !review.fetch("reason").strip.empty?
              raise ArgumentError, "Invalid review decision"
            end
            row.merge("review" => review, "agreement" => target == row.fetch("target").to_s,
              "excluded" => target == "exclude", "relabelled" => target != "exclude" && target != row.fetch("target").to_s)
          end
          cells = decisions.group_by { |row| row.fetch("source") }.transform_values do |items|
            { "count" => items.size, "excluded" => items.count { |row| row.fetch("excluded") },
              "relabelled" => items.count { |row| row.fetch("relabelled") }, "agreement" => items.count { |row| row.fetch("agreement") }.fdiv(items.size) }
          end
          { "decisions" => decisions, "by_source" => cells, "reviewer_scope" => "Record actual reviewers and limitations; no automatic semantic adjudication" }
        end

        def self.integrity!(rows)
          raise ArgumentError, "Duplicate decision IDs" unless rows.map { |row| row.fetch("id") }.uniq.size == rows.size
          inputs = rows.group_by { |row| [row.fetch("state"), row.fetch("question"), row.fetch("options")] }
          raise ArgumentError, "Conflicting gold on identical input" if inputs.values.any? { |items| items.map { |row| row.fetch("target") }.uniq.size > 1 }
          rows.each do |row|
            Example.new(row)
            next unless row.dig("world", "version") == 4
            world = row.fetch("world")
            audit_rendering!(row, world)
            values = world.fetch("facts").select { |fact| fact.fetch("actor") == world.fetch("actor") && fact.fetch("event") == world.fetch("query_event") }
            raise ArgumentError, "Conflicting explicit facts" if values.map { |fact| fact.fetch("truth") }.uniq.size > 1
            expected = values.empty? ? "unknown" : values.first.fetch("truth") == world.fetch("assertion") ? "yes" : "no"
            raise ArgumentError, "Gold disagrees with explicit facts" unless row.fetch("target") == expected
            unless world.fetch("row_truth_pattern") == world.fetch("facts").first(2).map { |fact| fact.fetch("truth") }
              raise ArgumentError, "Row truth metadata disagrees with facts"
            end
          end
          true
        end

        def self.audit_rendering!(row, world)
          chinese = row.fetch("language") == "zh-CN"
          raise ArgumentError, "Nonboolean assertion" unless [true, false].include?(world.fetch("assertion"))
          expected_mixed = world.fetch("facts").size == 2 && world.fetch("facts").map { |fact| fact.fetch("truth") }.uniq.size == 2
          raise ArgumentError, "Mixed-truth metadata disagrees" unless world.fetch("mixed") == expected_mixed
          clause = lambda do |actor, event, truth|
            phrase = FactualContrasts::EVENTS.fetch(event).fetch((chinese ? 0 : 2) + (truth ? 0 : 1))
            chinese ? "#{actor}#{phrase}" : "#{actor} #{phrase}"
          end
          facts = world.fetch("facts").map do |fact|
            raise ArgumentError, "Nonboolean fact" unless [true, false].include?(fact.fetch("truth"))
            clause.call(fact.fetch("actor"), fact.fetch("event"), fact.fetch("truth"))
          end
          states = [facts, facts.reverse].map { |values| values.join(chinese ? "。" : ". ") + (chinese ? "。" : ".") }
          raise ArgumentError, "Rendered facts disagree with metadata" unless states.include?(row.fetch("state"))
          claim = clause.call(world.fetch("actor"), world.fetch("query_event"), world.fetch("assertion"))
          question = chinese ? "根据记录判断陈述：#{claim}。" : "Judge this claim against the record: #{claim}."
          raise ArgumentError, "Rendered query disagrees with metadata" unless row.fetch("question") == question
        end

        def self.assert_disjoint!(*panels)
          groups = panels.map { |rows| rows.map { |row| row.fetch("group_id") }.to_set }
          materials = panels.map { |rows| rows.map { |row| digest(row.fetch("state")) }.to_set }
          raise ArgumentError, "Family leakage" if groups.combination(2).any? { |a, b| !(a & b).empty? }
          raise ArgumentError, "Record material leakage" if materials.combination(2).any? { |a, b| !(a & b).empty? }
          true
        end

        # Retrospective metadata review only; this is not a legacy inference path.
        def self.legacy_integrity_report(rows)
          controlled = rows.select { |row| row.dig("world", "version") == 3 }
          incorrect_gold = controlled.count do |row|
            world = row.fetch("world")
            facts = world.fetch("facts")
            actor = world.fetch("queried_actor")
            expected = facts.key?(actor) ? (facts.fetch(actor) == world.fetch("assertion") ? "yes" : "no") : "unknown"
            row.fetch("target") != expected
          end
          stale = controlled.count { |row| row.dig("world", "truth_pattern") != row.dig("world", "facts").values.first(2) }
          { "rows" => controlled.size, "families" => controlled.map { |row| row.fetch("group_id") }.uniq.size,
            "gold_mismatches" => incorrect_gold, "stale_truth_pattern" => stale,
            "unknown_kinds" => { "absent_actor" => controlled.count { |row| row.fetch("target") == "unknown" } },
            "scope" => "Independent gold derivation from stored explicit facts across complete v03 families; sampled rendered text reviewed separately. Stale metadata is not itself a label error." }
        end

        def self.digest(value)
          Digest::SHA256.hexdigest(value)
        end
      end
    end
  end
end
