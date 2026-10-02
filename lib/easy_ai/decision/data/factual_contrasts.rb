module EasyAI
  module Decision
    module Data
      # Gold labels follow explicit recorded facts. No keyword labels or inference rules.
      class FactualContrasts
        EVENTS = {
          "transport" => ["买了车票", "没有买车票", "bought a train ticket", "did not buy a train ticket"],
          "shopping" => ["买了外套", "没有买外套", "bought a coat", "did not buy a coat"],
          "work" => ["参加了会议", "没有参加会议", "attended the meeting", "did not attend the meeting"],
          "education" => ["交了作业", "没有交作业", "submitted the assignment", "did not submit the assignment"],
          "household" => ["关了窗户", "没有关窗户", "closed the window", "did not close the window"],
          "delivery" => ["收到了包裹", "没有收到包裹", "received the parcel", "did not receive the parcel"],
          "leisure" => ["看了电影", "没有看电影", "watched the film", "did not watch the film"],
          "communication" => ["发了邮件", "没有发邮件", "sent the email", "did not send the email"],
          "library" => ["借了书", "没有借书", "borrowed the book", "did not borrow the book"],
          "garden" => ["浇了花", "没有浇花", "watered the flowers", "did not water the flowers"],
          "sports" => ["完成了比赛", "没有完成比赛", "finished the race", "did not finish the race"],
          "finance" => ["付了账单", "没有付账单", "paid the bill", "did not pay the bill"]
        }.freeze
        TRAIN_EVENTS = EVENTS.keys.first(8).freeze
        LANGUAGES = %w[en-US zh-CN].freeze
        NAMES = {
          "en-US" => %w[Alex Sam Robin Morgan Taylor Casey Jordan Avery Quinn Riley Jamie Drew],
          "zh-CN" => %w[小陈 小周 小林 小吴 小赵 小许 小王 小李 小郑 小刘 小孙 小何]
        }.freeze

        def self.splits
          result = %w[train validation calibration test].to_h { |split| [split, []] }
          EVENTS.each_key do |event|
            allocation = TRAIN_EVENTS.include?(event) ? { "train" => 0...80, "validation" => 80...84, "calibration" => 84...88, "test" => 88...108 } : { "test" => 0...40 }
            allocation.each do |split, identities|
              identities.each do |identity|
                LANGUAGES.each do |language|
                  result.fetch(split).concat(new(event, identity, language).rows)
                end
              end
            end
          end
          result
        end

        def initialize(event, identity, language)
          @event, @identity, @language = event, identity, language
          @group = "factual-v02:#{event}:#{identity}"
          names = NAMES.fetch(language)
          @actors = [0, 5, 9].map { |offset| "#{names.fetch((identity + offset) % names.size)}#{identity}" }
        end

        def rows
          pairs = (0...4).flat_map do |variant|
            [true, false].map { |truth| build(variant, truth) }
          end
          pairs + [unknown]
        end

        private

        def clause(actor, truth)
          text = EVENTS.fetch(@event).fetch((@language == "zh-CN" ? 0 : 2) + (truth ? 0 : 1))
          @language == "zh-CN" ? "#{actor}#{text}。" : "#{actor} #{text}."
        end

        def question(variant)
          assertion = variant != 1
          claim = clause(queried_actor(variant), assertion)
          @language == "zh-CN" ? "根据记录，以下陈述成立吗：#{claim}" : "According to the record, is this statement true: #{claim}"
        end

        def options(variant)
          texts = if @language == "zh-CN"
            variant.even? ? ["成立", "不成立", "信息不足，无法判断"] : ["是", "不是", "无法确定"]
                  else
            variant.even? ? ["true", "false", "not enough information"] : ["yes", "no", "cannot determine"]
                  end
          %w[yes no unknown].zip(texts).map { |id, text| { "id" => id, "text" => text } }
        end

        def build(variant, truth)
          facts = [[@actors.first, truth]]
          facts << [@actors[1], !truth] if variant >= 2
          facts << [@actors[2], false] if variant == 3
          rendered = variant == 3 ? facts.reverse : facts
          target = facts.to_h.fetch(queried_actor(variant)) == (variant != 1) ? "yes" : "no"
          row(variant, facts, rendered.map { |actor, value| clause(actor, value) }.join(" "),
            target, source: "Factual-Contrast")
            .merge("contrast_groups" => { "fact_flip" => "#{@group}:#{@language}:#{variant}" })
        end

        def unknown
          facts = [[@actors[1], true], [@actors[2], false]]
          row(4, facts, facts.map { |actor, value| clause(actor, value) }.join(" "), "unknown", source: "Factual-Uncertainty")
        end

        def queried_actor(variant)
          @actors.fetch(variant == 3 ? 1 : 0)
        end

        def row(variant, facts, state, target, source:)
          { "id" => "#{@group}:#{@language}:#{variant}:#{target}", "group_id" => @group, "source" => source,
            "language" => @language, "state" => state, "question" => question(variant), "options" => options(variant), "target" => target,
            "world" => { "event" => @event, "facts" => facts.to_h, "queried_actor" => queried_actor(variant),
              "assertion" => variant != 1, "variant" => variant,
              "phenomenon" => %w[fact_flip negative_question actor_binding irrelevant_fact uncertainty].fetch(variant),
              "unseen_event" => !TRAIN_EVENTS.include?(@event) } }
        end
      end
    end
  end
end
