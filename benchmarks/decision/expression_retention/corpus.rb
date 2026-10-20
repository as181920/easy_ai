module DecisionExpressionRetention
  module Corpus
    module_function

    def gold!(row)
      DecisionEventGeneralization::Corpus.gold!(row)
    end

    def world_rows(actors, events, language:, group:, form: 0, axis: :actor)
      # The earlier generator supplies complete truth/query/order units; only wording changes.
      safe_events = %w[work shopping]
      base = DecisionEventGeneralization::Corpus.world_rows(actors, safe_events, language: language, group: group, axis: axis)
      base.map do |row|
        mapping = safe_events.zip(events).to_h
        world = row.fetch("world")
        facts = world.fetch("facts").map { |fact| fact.merge("event" => mapping.fetch(fact.fetch("event"))) }
        revised = row.merge("source" => row.fetch("source").sub("V08", "V09"), "world" => world.merge("generator" => "expression-v09", "family_id" => group))
        rendered = Catalog.render(revised, facts: facts, event: mapping.fetch(world.fetch("query_event")), form: form)
        rendered.merge("id" => "#{rendered.fetch('id')}:form#{form}", "group_id" => "#{group}:form#{form}",
          "contrast_groups" => rendered.fetch("contrast_groups").transform_values { |value| "#{value}:form#{form}" })
      end
    end

    def panels(split, worlds:, excluded_families:, historical_materials:)
      raise ArgumentError, "Even world count required" unless worlds.positive? && worlds.even?
      used = excluded_families.dup
      Catalog::VIEWS.to_h do |view|
        values = Catalog::LANGUAGES.flat_map do |language|
          rng = Random.new(Digest::SHA256.hexdigest("v09:#{split}:#{view}:#{language}").to_i(16) % (2**31))
          names = %w[actors composed].include?(view) ? Catalog.novel_names(language, split) : EasyAI::Decision::Data::ScalableJudgmentCorpus.actors(language, "canonical")
          events = %w[events composed].include?(view) ? Catalog::POOLS.fetch(split.to_s) : EasyAI::Decision::Data::FactualContrasts::TRAIN_EVENTS
          forms = %w[expressions composed].include?(view) ? Catalog::FORMS.fetch(split.to_s) : [0]
          worlds.times.flat_map do |index|
            selected = nil
            20_000.times do
              actors = names.sample(2, random: rng)
              actions = events.sample(2, random: rng)
              rows = forms.flat_map { |form| world_rows(actors, actions, language: language, group: "v09:#{split}:#{view}:#{index}", form: form, axis: index.even? ? :actor : :event) }
              family = Catalog.family(rows.first)
              next if used.include?(family) || rows.any? { |row| historical_materials.include?(Catalog.material(row.fetch("state"))) }
              used << family
              selected = rows
              break
            end
            raise "Finite v09 world allocation exhausted" unless selected
            selected
          end
        end
        [view, values]
      end
    end
  end
end
