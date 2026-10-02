module EasyAI
  module Decision
    module Data
      # Render explicit worlds; labels never depend on recognizing words in the text.
      class RobustFacts
        EVENTS = FactualContrasts::EVENTS
        LANGUAGES = FactualContrasts::LANGUAGES
        TRAIN_EVENTS = FactualContrasts::TRAIN_EVENTS
        FAMILY_VERSION = "factual-v03-r2".freeze
        OPTIONS = {
          "en-US" => [["true", "false", "not enough information"], ["supported", "contradicted", "insufficient evidence"],
            ["correct", "incorrect", "cannot determine from the record"], ["the statement holds", "the statement does not hold", "the record does not establish an answer"],
            ["the claim agrees with the facts", "the claim conflicts with the facts", "there is not enough evidence to decide"],
            ["the claim is supported by the evidence", "the claim is contradicted by the evidence", "the evidence leaves the claim unresolved"]],
          "zh-CN" => [["成立", "不成立", "信息不足"], ["与记录一致", "与记录矛盾", "证据不足"],
            ["正确", "不正确", "根据记录无法确定"], ["这个说法成立", "这个说法不成立", "记录未提供足够依据"],
            ["陈述符合事实", "陈述与事实冲突", "没有足够证据作出判断"], ["证据支持这一陈述", "证据否定这一陈述", "证据无法确定这一陈述"]]
        }.freeze
        EN_FIRST = %w[Alex Sam Robin Morgan Taylor Casey Jordan Avery Quinn Riley Jamie Drew].freeze
        EN_LAST = %w[Chen Wilson Lee Brown Miller Davis Moore Clark Hall Young Allen King Scott Green Baker Adams Nelson Carter].freeze
        ZH_FIRST = %w[陈 周 林 吴 赵 许 王 李 郑 刘 孙 何 张 黄 杨 朱 胡 高 罗 梁].freeze
        ZH_LAST = %w[明 安 宁 平 晨 乐 宇 佳 青 兰 文 瑜 林 辰 夏 秋 雨 欣 然 可].freeze

        def self.splits
          result = %w[train validation calibration test].to_h { |split| [split, []] }
          EVENTS.each_key do |event|
            allocations = TRAIN_EVENTS.include?(event) ? { "train" => 32, "validation" => 4, "calibration" => 4, "test" => 15 } : { "test" => 20 }
            allocations.each do |split, count|
              count.times do |identity|
                LANGUAGES.each { |language| result.fetch(split).concat(new(event, identity, language, split: split).rows) }
              end
            end
          end
          result
        end

        def initialize(event, identity, language, split: "train")
          @event, @identity, @language, @split = event, identity, language, split
          @group = "#{FAMILY_VERSION}:#{split}:#{event}:#{identity}"
          rng = Random.new(Digest::SHA256.hexdigest(@group).to_i(16) % (2**31))
          first, last = language == "zh-CN" ? [ZH_FIRST, ZH_LAST] : [EN_FIRST, EN_LAST]
          @actors = first.product(last).map { |a, b| language == "zh-CN" ? a + b : "#{a} #{b}" }.sample(3, random: rng)
          @facts = @actors.first(2).zip([identity.even?, (identity / 2).even?]).to_h
          @style = split == "train" ? rng.rand(2) : { "validation" => 2, "calibration" => 2, "test" => 3 }.fetch(split)
          @wording = split == "train" ? rng.rand(2) : split == "test" ? 4 : 2
          @irrelevant_truth = rng.rand(2).zero?
        end

        def rows
          result = [0, 1].flat_map do |order|
            [0, 1].flat_map do |actor|
              [true, false].map { |assertion| known(order, actor, assertion) }
            end
          end
          [0, 1].each do |order|
            [true, false].each { |assertion| result << unknown(order, assertion) }
          end
          [0, 1].each do |actor|
            [true, false].each do |assertion|
              result << known(0, actor, assertion, variant: "wording")
              result << known(0, actor, assertion, variant: "irrelevant")
            end
          end
          [true, false].each { |assertion| result << unknown(0, assertion, variant: "wording") }
          [true, false].each do |assertion|
            [true, false].each { |truth| result << fact_flip(assertion, truth) }
          end
          result
        end

        private

        def clause(actor, truth)
          phrase = EVENTS.fetch(@event).fetch((@language == "zh-CN" ? 0 : 2) + (truth ? 0 : 1))
          if @language == "zh-CN"
            ["#{actor}#{phrase}。", "#{actor}确实#{phrase}。", "记录注明：#{actor}#{phrase}。", "关于#{actor}，记载的情况是#{phrase}。"].fetch(@style)
          else
            ["#{actor} #{phrase}.", "The record says that #{actor} #{phrase}.", "It is recorded that #{actor} #{phrase}.", "For #{actor}, the recorded fact is that they #{phrase}."].fetch(@style)
          end
        end

        def question(actor, assertion)
          phrase = EVENTS.fetch(@event).fetch((@language == "zh-CN" ? 0 : 2) + (assertion ? 0 : 1))
          claim = @language == "zh-CN" ? "#{actor}#{phrase}" : "#{actor} #{phrase}"
          if @language == "zh-CN"
            ["以下说法成立吗：#{claim}？", "根据记录判断：#{claim}。这个说法是否成立？", "记录是否支持这个说法：#{claim}？", "请判断这一陈述是否符合记录：#{claim}。"].fetch(@style)
          else
            ["Is this statement true: #{claim}?", "Does the record support this statement: #{claim}?", "Based on the record, is the following assertion correct: #{claim}?", "Decide whether this claim agrees with the record: #{claim}."].fetch(@style)
          end
        end

        def known(order, actor, assertion, variant: "base")
          facts = @facts.dup
          facts[@actors.last] = @irrelevant_truth if variant == "irrelevant"
          row(facts, @actors.fetch(actor), assertion, order, variant)
        end

        def unknown(order, assertion, variant: "base")
          row(@facts, @actors.last, assertion, order, variant)
        end

        def fact_flip(assertion, truth)
          row(@facts.merge(@actors.first => truth), @actors.first, assertion, 0, "fact_flip")
        end

        def row(facts, actor, assertion, order, variant)
          target = facts.key?(actor) ? (facts.fetch(actor) == assertion ? "yes" : "no") : "unknown"
          wording = variant == "wording" ? @wording + 1 : @wording
          # Train-only alternates stay inside training vocabulary; held-out wording is never admitted here.
          wording = 1 - @wording if @split == "train" && variant == "wording"
          options = %w[yes no unknown].zip(OPTIONS.fetch(@language).fetch(wording)).map { |id, text| { "id" => id, "text" => text } }
          options.pop if variant == "wording" && target != "unknown"
          rendered = order.zero? ? facts.to_a : facts.to_a.reverse
          id = "#{@group}:#{@language}:#{actor}:#{assertion}:#{order}:#{variant}:#{variant == 'fact_flip' ? facts.fetch(actor) : 'fixed'}"
          { "id" => id, "group_id" => @group, "source" => target == "unknown" ? "Factual-V03-Unknown" : "Factual-V03-Known",
            "language" => @language, "state" => rendered.map { |name, truth| clause(name, truth) }.join(" "),
            "question" => question(actor, assertion), "options" => options, "target" => target,
            "contrast_groups" => contrasts(actor, assertion, order, variant, target),
            "world" => { "version" => 3, "event" => @event, "facts" => facts, "queried_actor" => actor, "assertion" => assertion,
              "fact_order" => rendered.map(&:first), "queried_position" => rendered.index { |name, _| name == actor },
              "truth_pattern" => @facts.values, "expression" => @style, "candidate_wording" => wording, "variant" => variant,
              "phenomenon" => target == "unknown" ? "uncertainty" : variant == "base" ? "actor_binding" : variant,
              "unseen_event" => !TRAIN_EVENTS.include?(@event) } }
        end

        def contrasts(actor, assertion, order, variant, target)
          prefix = "#{@group}:#{@language}"
          return { "fact_flip" => "#{prefix}:#{assertion}" } if variant == "fact_flip"
          result = { "question_flip" => "#{prefix}:#{actor}:#{order}:#{variant}" }
          if variant == "base"
            result["order"] = "#{prefix}:#{actor}:#{assertion}"
            result["actor_switch"] = "#{prefix}:#{assertion}:#{order}" unless target == "unknown"
          end
          if order.zero?
            result["wording"] = "#{prefix}:#{actor}:#{assertion}" if %w[base wording].include?(variant)
            result["irrelevant"] = "#{prefix}:#{actor}:#{assertion}" if %w[base irrelevant].include?(variant) && target != "unknown"
          end
          result
        end
      end
    end
  end
end
