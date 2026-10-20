module DecisionExpressionRetention
  module Catalog
    module_function

    NEW_EVENTS = {
      "button" => ["按下了按钮", "没有按下按钮", "pressed the button", "did not press the button"],
      "letter" => ["寄出了信", "没有寄出信", "mailed the letter", "did not mail the letter"],
      "pencil" => ["削了铅笔", "没有削铅笔", "sharpened the pencil", "did not sharpen the pencil"],
      "clock" => ["调整了时钟", "没有调整时钟", "adjusted the clock", "did not adjust the clock"],
      "plant" => ["给盆栽浇了水", "没有给盆栽浇水", "watered the plant", "did not water the plant"],
      "floor" => ["拖了地板", "没有拖地板", "mopped the floor", "did not mop the floor"],
      "boots" => ["擦了靴子", "没有擦靴子", "polished the boots", "did not polish the boots"],
      "cake" => ["切了蛋糕", "没有切蛋糕", "sliced the cake", "did not slice the cake"],
      "drawer" => ["清空了抽屉", "没有清空抽屉", "emptied the drawer", "did not empty the drawer"],
      "bicycle" => ["给自行车打了气", "没有给自行车打气", "inflated the bicycle tires", "did not inflate the bicycle tires"],
      "soup" => ["搅拌了汤", "没有搅拌汤", "stirred the soup", "did not stir the soup"],
      "map" => ["展开了地图", "没有展开地图", "unfolded the map", "did not unfold the map"],
      "basket" => ["提起了篮子", "没有提起篮子", "lifted the basket", "did not lift the basket"],
      "scarf" => ["织了围巾", "没有织围巾", "knitted the scarf", "did not knit the scarf"],
      "carrot" => ["切碎了胡萝卜", "没有切碎胡萝卜", "chopped the carrot", "did not chop the carrot"],
      "bell" => ["摇响了铃铛", "没有摇响铃铛", "rang the bell", "did not ring the bell"],
      "envelope" => ["封好了信封", "没有封好信封", "sealed the envelope", "did not seal the envelope"],
      "ribbon" => ["系好了丝带", "没有系好丝带", "fastened the ribbon", "did not fasten the ribbon"],
      "piano" => ["弹了钢琴", "没有弹钢琴", "played the piano", "did not play the piano"],
      "curtain" => ["拉开了窗帘", "没有拉开窗帘", "drew the curtain open", "did not draw the curtain open"],
      "rice" => ["煮了米饭", "没有煮米饭", "cooked the rice", "did not cook the rice"],
      "roof" => ["检查了屋顶", "没有检查屋顶", "inspected the roof", "did not inspect the roof"],
      "stamp" => ["贴好了邮票", "没有贴好邮票", "affixed the stamp", "did not affix the stamp"],
      "shelf" => ["组装了架子", "没有组装架子", "assembled the shelf", "did not assemble the shelf"]
    }.freeze
    EVENTS = DecisionEventGeneralization::Catalog::EVENTS.merge(NEW_EVENTS).freeze
    POOLS = { "train" => NEW_EVENTS.keys.first(8), "development" => NEW_EVENTS.keys.drop(8).first(8), "lockbox" => NEW_EVENTS.keys.drop(16) }.freeze
    FORMS = { "train" => [1, 2, 7, 8], "development" => [9, 10], "lockbox" => [11, 12] }.freeze
    LANGUAGES = %w[en-US zh-CN].freeze
    VIEWS = %w[combinations actors events expressions composed].freeze

    def phrase(event, truth, language)
      values = EVENTS.fetch(event) { EasyAI::Decision::Data::FactualContrasts::EVENTS.fetch(event) }
      values.fetch((language.to_s == "zh-CN" ? 0 : 2) + (truth ? 0 : 1))
    end

    def clause(actor, event, truth, language, form)
      action = phrase(event, truth, language)
      claim = language.to_s == "zh-CN" ? "#{actor}#{action}" : "#{actor} #{action}"
      forms = if language.to_s == "zh-CN"
        [claim, "记录：#{claim}", "日志显示，#{claim}", "记录中写道，#{claim}", "#{actor}：#{action}", "#{claim}，这是日志记载的内容", "关于#{actor}，日志记录其#{action}",
          "观察记录：#{claim}", "日志列出了这一事实：#{claim}", "#{claim}；这一点记载在记录中", "记录的事实是，#{claim}", "已记录的观察结果：#{claim}", "#{claim}（根据记录的事实）"]
              else
        [claim, "Record: #{claim}", "According to the log, #{claim}", "The record states that #{claim}", "#{actor}: #{action}", "#{claim}, as recorded in the log", "Regarding #{actor}, the log records that they #{action}",
          "Observed entry: #{claim}", "The log lists this fact: #{claim}", "#{claim}; this is stated in the record", "The recorded fact is that #{claim}", "Recorded observation: #{claim}", "#{claim} (according to the recorded facts)"]
              end
      forms.fetch(form)
    end

    def question(actor, event, assertion, language, form)
      claim = clause(actor, event, assertion, language, 0)
      forms = if language.to_s == "zh-CN"
        ["根据记录判断陈述：#{claim}。", "记录是否支持这句话：#{claim}？", "请用日志核对：#{claim}。", "仅根据记录判断：#{claim}。", "关于#{actor}的说法是否成立：#{phrase(event, assertion, language)}？",
          "根据日志，判断#{claim}是否属实。", "日志能否证实下面的说法：#{claim}？", "待判断的说法是：#{claim}。它是否符合日志？", "根据记录的事实，这个说法是否正确：#{claim}？",
          "根据已记录的事实，#{claim}这一说法是否属实？", "请判断日志是否支持这个说法：#{claim}。", "用记录的事实核对这个说法：#{claim}。", "有人提出说法：#{claim}。根据日志，该说法是否正确？"]
              else
        ["Judge this claim against the record: #{claim}.", "Is this statement supported by the record: #{claim}?", "Check this statement using the log: #{claim}.", "Using only the record, judge: #{claim}.", "Statement about #{actor}: #{phrase(event, assertion, language)}. Is it supported?",
          "Based on the log, decide whether #{claim}.", "Does the log establish the following claim: #{claim}?", "The claim to assess is: #{claim}. Does it agree with the log?", "From the recorded facts, is this claim correct: #{claim}?",
          "According to the recorded facts, is it true that #{claim}?", "Assess whether the log supports this claim: #{claim}.", "Check the claim #{claim} against the recorded facts.", "The proposed claim is: #{claim}. Is the claim correct given the log?"]
              end
      forms.fetch(form)
    end

    def render(row, facts: row.fetch("world").fetch("facts"), actor: row.fetch("world").fetch("actor"), event: row.fetch("world").fetch("query_event"), form: 0)
      world = row.fetch("world")
      language = row.fetch("language")
      ordered = world.fetch("render_order", row.fetch("id").split(":").last.to_i).zero? ? facts : facts.reverse
      delimiter = language == "zh-CN" ? "。" : ". "
      state = ordered.map { |fact| clause(fact.fetch("actor"), fact.fetch("event"), fact.fetch("truth"), language, form) }.join(delimiter) + (language == "zh-CN" ? "。" : ".")
      row.merge("state" => state, "question" => question(actor, event, world.fetch("assertion"), language, form),
        "world" => world.merge("facts" => facts, "actor" => actor, "event" => event, "query_event" => event, "form" => form))
    end

    def family(row)
      DecisionEventGeneralization::Catalog.family(row)
    end

    def material(state)
      DecisionEventGeneralization::Catalog.material(state)
    end

    def novel_names(language, split)
      names = if language.to_s == "zh-CN"
        %w[蒲 殷 郜 厉 裘 巩 冉 蓟].product(%w[沐禾 书遥 云舒 南笙 清晏 闻溪 怀瑾 令仪]).map(&:join)
              else
        %w[Aurelia Beatrix Cecily Delphine Elowen Felicity Gwendolen Hermione].product(%w[Whitaker Prescott Langley Ashford Redmond Winslow Hartwell Ellington]).map { |a, b| "#{a} #{b}" }
              end
      split.to_s == "development" ? names.first(32) : names.last(32)
    end
  end
end
