module EasyAI
  module Decision
    module Data
      # This oracle generates supervision. Its facts/targets are never neural inputs.
      class JudgmentCorpus
        LANGUAGES = %w[en-US zh-CN].freeze
        OPTIONS = { "en-US" => ["supported by the record", "contradicted by the record", "insufficient evidence in the record"],
          "zh-CN" => ["记录支持该陈述", "记录否定该陈述", "记录信息不足，无法判断"] }.freeze

        def self.options(language, binary: false, held: false)
          texts = held ? RobustFacts::OPTIONS.fetch(language).last : OPTIONS.fetch(language)
          %w[yes no unknown].zip(texts).first(binary ? 2 : 3).map { |id, text| { "id" => id, "text" => text } }
        end

        def self.splits(counts: { "train" => 32, "validation" => 100, "calibration" => 40, "test" => 200 })
          counts.to_h do |split, count|
            rows = count.times.flat_map { |index| LANGUAGES.flat_map { |language| new(split, index, language).rows } }
            [split, rows]
          end
        end

        def initialize(split, index, language)
          @split, @index, @language = split, index, language
          @group = "judgment-v04:#{split}:#{index}"
          rng = Random.new(QualityAudit.digest(@group).to_i(16) % (2**31))
          pool = language == "zh-CN" ? RobustFacts::ZH_FIRST.product(RobustFacts::ZH_LAST).map(&:join) :
            RobustFacts::EN_FIRST.product(RobustFacts::EN_LAST).map { |first, last| "#{first} #{last}" }
          # Reserve actual actor/event combinations, not only split-prefixed IDs.
          ordinal = { "train" => 0, "validation" => 32, "calibration" => 132, "test" => 172 }.fetch(split) + index
          first = pool.shuffle(random: Random.new(404)).fetch(ordinal % pool.size)
          @actors = [first, *pool.reject { |actor| actor == first }.sample(2, random: rng)]
          keys = FactualContrasts::TRAIN_EVENTS
          event_index = ordinal + ordinal / pool.size
          @events = [keys[event_index % keys.size], keys[(event_index + 3) % keys.size]]
        end

        def rows
          result = []
          [false, true].product([false, true]).each_with_index do |truths, assignment|
            # Same actor/different events alternates with different actors/same event.
            actors = @index.odd? ? [@actors.first, @actors.first] : @actors.first(2)
            events = @index.odd? ? @events : [@events.first, @events.first]
            facts = 2.times.map { |i| { "actor" => actors[i], "event" => events[i], "truth" => truths[i] } }
            orders = @split == "train" ? [0, 1] : [0]
            orders.each do |order|
              [true, false].each do |assertion|
                queries = facts.map { |fact| [fact.fetch("actor"), fact.fetch("event"), "known"] }
                # Present actor, unrecorded event; absent actor, recorded event.
                missing = @index.odd? ? (FactualContrasts::TRAIN_EVENTS - @events).first : @events.last
                queries += [[@actors.first, missing, "missing_event"], [@actors.last, @events.first, "absent_actor"]]
                queries.each_with_index do |(actor, event, kind), query|
                  result << row(facts, actor, event, assertion, order, assignment, query, "joint", kind)
                  result << row(facts, actor, event, assertion, order, assignment, query, "binary", kind) if kind == "known"
                end
              end
            end
            if @split == "train"
              facts.each_with_index do |fact, query|
                [true, false].each do |assertion|
                  result << row([fact], fact.fetch("actor"), fact.fetch("event"), assertion, 0, assignment, query, "single", "known")
                end
              end
            end
          end
          result
        end

        private

        def clause(actor, event, truth)
          phrase = FactualContrasts::EVENTS.fetch(event).fetch((@language == "zh-CN" ? 0 : 2) + (truth ? 0 : 1))
          @language == "zh-CN" ? "#{actor}#{phrase}" : "#{actor} #{phrase}"
        end

        def row(facts, actor, event, assertion, order, assignment, query, profile, kind)
          evidence = facts.find { |fact| fact.fetch("actor") == actor && fact.fetch("event") == event }
          target = evidence ? (evidence.fetch("truth") == assertion ? "yes" : "no") : "unknown"
          claim = clause(actor, event, assertion)
          rendered = order.zero? ? facts : facts.reverse
          state = rendered.map { |fact| clause(fact.fetch("actor"), fact.fetch("event"), fact.fetch("truth")) }.join(@language == "zh-CN" ? "。" : ". ")
          state += @language == "zh-CN" ? "。" : "."
          prefix = "#{@group}:#{@language}:#{profile}"
          axes = { "question_flip" => "#{prefix}:#{assignment}:#{query}:#{order}" }
          axes["binding"] = "#{prefix}:#{assignment}:#{assertion}:#{order}" if kind == "known" && profile != "single"
          unless profile == "single" || kind != "known"
            # Hold the other fact fixed, and flip either queried actor/event fact.
            other = facts.fetch(1 - query).fetch("truth")
            axes["fact_flip"] = "#{prefix}:#{query}:#{assertion}:#{order}:#{other}"
          end
          { "id" => "#{prefix}:#{assignment}:#{query}:#{assertion}:#{order}", "group_id" => @group,
            "language" => @language, "source" => profile == "single" ? "Factual-V04-Single" : "Factual-V04", "state" => state,
            "question" => @language == "zh-CN" ? "根据记录判断陈述：#{claim}。" : "Judge this claim against the record: #{claim}.",
            "options" => self.class.options(@language, binary: profile != "joint"), "target" => target, "contrast_groups" => axes,
            "world" => { "version" => 4, "facts" => facts, "actor" => actor, "query_event" => event, "event" => event,
              "assertion" => assertion, "row_truth_pattern" => facts.first(2).map { |fact| fact.fetch("truth") },
              "profile" => profile, "kind" => kind, "assignment" => assignment, "query" => query,
              "mixed" => facts.size == 2 && facts.map { |fact| fact.fetch("truth") }.uniq.size == 2 } }
        end
      end
    end
  end
end
