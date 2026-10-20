module EasyAI
  module Decision
    module Data
      # Allocate complete relational worlds from finite, collision-free catalogs.
      # Truth assignments, claims and labels still use the audited v04 renderer.
      class ScalableJudgmentCorpus < JudgmentCorpus
        PANELS = %w[canonical novel_actor novel_event].freeze
        SHARED_ACTORS = 128
        EN_FIRST = %w[Adrian Blair Cameron Devon Ellis Finley Hayden Kendall Logan Parker Reese Sidney].freeze
        EN_LAST = %w[Brooks Cooper Evans Fisher Gray Howard Jackson Lewis Martin Perry Reed Rogers Shaw Turner Walker Ward Watson Wright].freeze
        ZH_FIRST = %w[顾 萧 谢 韩 唐 方 曹 袁 邵 陆 崔 邱 蒋 蔡 汪 田 丁 姚 杜 戴].freeze

        def self.actors(language, panel)
          first, last = language == "zh-CN" ? [ZH_FIRST, RobustFacts::ZH_LAST] : [EN_FIRST, EN_LAST]
          pool = first.product(last).map { |a, b| language == "zh-CN" ? a + b : "#{a} #{b}" }
          panel == "novel_actor" ? pool.drop(SHARED_ACTORS) : pool.first(SHARED_ACTORS)
        end

        def self.catalog(language, panel, kind)
          @catalogs ||= {}
          @catalogs[[language, panel, kind]] ||= begin
            pool = actors(language, panel)
            events = panel == "novel_event" ? FactualContrasts::EVENTS.keys - FactualContrasts::TRAIN_EVENTS : FactualContrasts::TRAIN_EVENTS
            values = if kind.zero?
              pool.combination(2).flat_map do |a, b|
                events.map { |event| [world_actors(pool, a, b, "#{a}:#{b}:#{event}"), [event, events[(events.index(event) + 1) % events.size]]] }
              end
                     else
              pool.flat_map do |a|
                others = pool.reject { |name| name == a }.first(2)
                events.combination(2).map { |pair| [world_actors(pool, a, others.first, "#{a}:#{pair.join(':')}"), pair] }
              end
                     end
            values.shuffle(random: Random.new(QualityAudit.digest("scale-v06:#{language}:#{panel}:#{kind}").to_i(16) % (2**31))).freeze
          end
        end

        def self.world_actors(pool, a, b, key)
          index = QualityAudit.digest("absent:#{key}").to_i(16) % pool.size
          index = (index + 1) % pool.size while [a, b].include?(pool.fetch(index))
          [a, b, pool.fetch(index)]
        end

        def initialize(split, index, language, counts:, panel: "canonical")
          @split, @index, @language = split.to_s, index, language.to_s
          panel = panel.to_s
          @panel = panel
          raise ArgumentError, "Unknown scaling panel or language" unless PANELS.include?(panel) && LANGUAGES.include?(@language)
          normalized = counts.to_h { |key, value| [key.to_s, value] }
          offsets = self.class.split_offsets(counts)
          unless index.is_a?(Integer) && index >= 0 && index < normalized.fetch(@split)
            raise ArgumentError, "World index outside reserved split"
          end
          ordinal = offsets.fetch(@split) + index
          # Split sizes need not be even: the global ordinal determines the type.
          @index = ordinal
          @group = "scale-v06:#{panel}:#{@split}:#{index}"
          catalog = self.class.catalog(@language, panel, ordinal % 2)
          raise ArgumentError, "Requested split exceeds unique world capacity" unless ordinal / 2 < catalog.size
          @actors, @events = catalog.fetch(ordinal / 2)
        end

        private

        def available_events
          @panel == "novel_event" ? FactualContrasts::EVENTS.keys - FactualContrasts::TRAIN_EVENTS : FactualContrasts::TRAIN_EVENTS
        end

        def row(...)
          result = super
          result["source"] = result.fetch("source").end_with?("Single") ? "Factual-V06-Single" : "Factual-V06-#{@index.even? ? 'Actor' : 'Event'}"
          result.fetch("world")["generator"] = "scale-v06"
          result
        end
      end
    end
  end
end
