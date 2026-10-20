module DecisionEventGeneralization
  module Corpus
    module_function

    def gold!(row)
      world = row.fetch("world")
      evidence = world.fetch("facts").find { |fact| fact.fetch("actor") == world.fetch("actor") && fact.fetch("event") == world.fetch("query_event") }
      expected = evidence && evidence.fetch("truth") == world.fetch("assertion") ? "yes" : "no"
      raise "Missing evidence in binary v08 row" unless evidence
      raise "Incorrect v08 gold" unless row.fetch("target") == expected
      expected
    end

    def world_rows(actors, events, language:, group:, form: 0, axis: :actor)
      axis = axis.to_s
      raise ArgumentError, "Invalid world axis" unless %w[actor event].include?(axis)
      identities = axis == "actor" ? actors.first(2) : [actors.first, actors.first]
      actions = axis == "actor" ? [events.first, events.first] : events.first(2)
      raise ArgumentError, "World must distinguish queried facts" unless identities.zip(actions).uniq.size == 2
      [false, true].repeated_permutation(2).flat_map.with_index do |truths, assignment|
        facts = 2.times.map { |index| { "actor" => identities[index], "event" => actions[index], "truth" => truths[index] } }
        [0, 1].flat_map do |order|
          [true, false].flat_map do |assertion|
            2.times.map do |query|
              prefix = "#{group}:#{language}:binary"
              row = { "id" => "#{prefix}:#{assignment}:#{query}:#{assertion}:#{order}", "group_id" => group, "language" => language.to_s,
                "source" => "Factual-V08-#{axis.capitalize}", "options" => EasyAI::Decision::Data::JudgmentCorpus.options(language.to_s, binary: true),
                "target" => truths[query] == assertion ? "yes" : "no",
                "contrast_groups" => { "binding" => "#{prefix}:#{assignment}:#{assertion}:#{order}", "question_flip" => "#{prefix}:#{assignment}:#{query}:#{order}",
                  "fact_flip" => "#{prefix}:#{query}:#{assertion}:#{order}:#{truths[1 - query]}" },
                "world" => { "version" => 4, "facts" => facts, "actor" => identities[query], "event" => actions[query], "query_event" => actions[query],
                  "query" => query, "assertion" => assertion, "profile" => "binary", "kind" => "known", "mixed" => truths.uniq.size == 2,
                  "assignment" => assignment, "render_order" => order, "generator" => "event-v08" } }
              Catalog.render(row, form: form)
            end
          end
        end
      end
    end

    def panels(split, worlds:, excluded_families:, historical_materials:)
      used = excluded_families.dup
      Catalog::VIEWS.to_h do |view|
        values = Catalog::LANGUAGES.flat_map do |language|
          rng = Random.new(Digest::SHA256.hexdigest("v08:#{split}:#{view}:#{language}").to_i(16) % (2**31))
          names = %w[actors composed].include?(view) ? Catalog.novel_names(language, split) : EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(language, "canonical")
          events = %w[events composed].include?(view) ? Catalog::POOLS.fetch(split.to_s) : EasyAI::Decision::Data::FactualContrasts::TRAIN_EVENTS
          worlds.times.flat_map do |index|
            selected = nil
            10_000.times do
              form = %w[expressions composed].include?(view) ? Catalog::FORMS.fetch(split.to_s).fetch(index % 2) : 0
              rows = world_rows(names.sample(2, random: rng), events.sample(2, random: rng), language: language, group: "v08:#{split}:#{view}:#{index}", form: form, axis: index.even? ? :actor : :event)
              family = Catalog.family(rows.first)
              next if used.include?(family) || rows.any? { |row| historical_materials.include?(Catalog.material(row.fetch("state"))) }
              used << family
              selected = rows
              break
            end
            raise "Finite v08 world allocation exhausted" unless selected
            selected
          end
        end
        [view, values]
      end
    end
  end
end
