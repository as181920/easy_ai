module DecisionEventGeneralization
  module Catalog
    module_function

    EVENTS = {
      "tea" => ["喝了茶", "没有喝茶", "drank the tea", "did not drink the tea"],
      "bottle" => ["打开了瓶子", "没有打开瓶子", "opened the bottle", "did not open the bottle"],
      "shirt" => ["洗了衬衫", "没有洗衬衫", "washed the shirt", "did not wash the shirt"],
      "phone" => ["给手机充了电", "没有给手机充电", "charged the phone", "did not charge the phone"],
      "gate" => ["锁了大门", "没有锁大门", "locked the gate", "did not lock the gate"],
      "photo" => ["打印了照片", "没有打印照片", "printed the photograph", "did not print the photograph"],
      "tree" => ["种了树", "没有种树", "planted the tree", "did not plant the tree"],
      "chair" => ["修了椅子", "没有修椅子", "repaired the chair", "did not repair the chair"],
      "cat" => ["喂了猫", "没有喂猫", "fed the cat", "did not feed the cat"],
      "wall" => ["粉刷了墙壁", "没有粉刷墙壁", "painted the wall", "did not paint the wall"],
      "bread" => ["烤了面包", "没有烤面包", "baked the bread", "did not bake the bread"],
      "blanket" => ["叠了毯子", "没有叠毯子", "folded the blanket", "did not fold the blanket"],
      "mirror" => ["擦了镜子", "没有擦镜子", "cleaned the mirror", "did not clean the mirror"],
      "bucket" => ["装满了水桶", "没有装满水桶", "filled the bucket", "did not fill the bucket"],
      "parcel" => ["捆好了包裹", "没有捆好包裹", "tied the parcel", "did not tie the parcel"],
      "candle" => ["点燃了蜡烛", "没有点燃蜡烛", "lit the candle", "did not light the candle"],
      "cup" => ["洗了杯子", "没有洗杯子", "washed the cup", "did not wash the cup"],
      "violin" => ["给小提琴调了音", "没有给小提琴调音", "tuned the violin", "did not tune the violin"],
      "bag" => ["收拾了背包", "没有收拾背包", "packed the bag", "did not pack the bag"],
      "apple" => ["称了苹果的重量", "没有称苹果的重量", "weighed the apple", "did not weigh the apple"],
      "orange" => ["剥了橙子", "没有剥橙子", "peeled the orange", "did not peel the orange"],
      "coat" => ["挂好了大衣", "没有挂好大衣", "hung up the coat", "did not hang up the coat"],
      "water" => ["烧开了水", "没有烧开水", "boiled the water", "did not boil the water"],
      "fence" => ["修好了围栏", "没有修好围栏", "mended the fence", "did not mend the fence"]
    }.freeze
    POOLS = { "train" => EVENTS.keys.first(8), "development" => EVENTS.keys.drop(8).first(8), "lockbox" => EVENTS.keys.drop(16) }.freeze
    FORMS = { "train" => [0, 1, 2], "development" => [3, 4], "lockbox" => [5, 6] }.freeze
    LANGUAGES = %w[en-US zh-CN].freeze
    VIEWS = %w[combinations actors events expressions composed].freeze

    def phrase(event, truth, language)
      phrases = EVENTS.fetch(event) { EasyAI::Decision::Data::FactualContrasts::EVENTS.fetch(event) }
      phrases.fetch((language.to_s == "zh-CN" ? 0 : 2) + (truth ? 0 : 1))
    end

    def clause(actor, event, truth, language, form = 0)
      action = phrase(event, truth, language)
      if language.to_s == "zh-CN"
        ["#{actor}#{action}", "记录：#{actor}#{action}", "日志显示，#{actor}#{action}", "记录中写道，#{actor}#{action}",
          "#{actor}：#{action}", "#{actor}#{action}，这是日志记载的内容", "关于#{actor}，日志记录其#{action}"].fetch(form)
      else
        ["#{actor} #{action}", "Record: #{actor} #{action}", "According to the log, #{actor} #{action}", "The record states that #{actor} #{action}",
          "#{actor}: #{action}", "#{actor} #{action}, as recorded in the log", "Regarding #{actor}, the log records that they #{action}"].fetch(form)
      end
    end

    def question(actor, event, assertion, language, form = 0)
      claim = clause(actor, event, assertion, language, 0)
      if language.to_s == "zh-CN"
        ["根据记录判断陈述：#{claim}。", "记录是否支持这句话：#{claim}？", "请用日志核对：#{claim}。", "仅根据记录判断：#{claim}。",
          "关于#{actor}的说法是否成立：#{phrase(event, assertion, language)}？", "根据日志，判断#{claim}是否属实。", "日志能否证实下面的说法：#{claim}？"].fetch(form)
      else
        ["Judge this claim against the record: #{claim}.", "Is this statement supported by the record: #{claim}?", "Check this statement using the log: #{claim}.",
          "Using only the record, judge: #{claim}.", "Statement about #{actor}: #{phrase(event, assertion, language)}. Is it supported?",
          "Based on the log, decide whether #{claim}.", "Does the log establish the following claim: #{claim}?"].fetch(form)
      end
    end

    def render(row, facts: row.fetch("world").fetch("facts"), actor: row.fetch("world").fetch("actor"), event: row.fetch("world").fetch("query_event"), form: 0)
      language = row.fetch("language")
      world = row.fetch("world")
      order = world.fetch("render_order", 0)
      ordered = order.zero? ? facts : facts.reverse
      delimiter = language == "zh-CN" ? "。" : ". "
      state = ordered.map { |fact| clause(fact.fetch("actor"), fact.fetch("event"), fact.fetch("truth"), language, form) }.join(delimiter) + (language == "zh-CN" ? "。" : ".")
      row.merge("state" => state, "question" => question(actor, event, world.fetch("assertion"), language, form),
        "world" => world.merge("facts" => facts, "actor" => actor, "event" => event, "query_event" => event, "form" => form))
    end

    def family(row)
      facts = row.fetch("world").fetch("facts")
      return nil unless facts.is_a?(Array) && facts.size == 2 && facts.all? { |fact| fact.is_a?(Hash) && fact.key?("actor") && fact.key?("event") }
      JSON.generate([row.fetch("language"), facts.map { |fact| [fact.fetch("actor"), fact.fetch("event")] }.sort])
    end

    def material(state)
      Digest::SHA256.hexdigest(state.downcase.gsub(/[[:space:][:punct:]。！？；，：]/, ""))
    end

    def novel_names(language, split)
      names = if language.to_s == "zh-CN"
        %w[阎 邢 邬 屠 苗 茅 甄 封].product(%w[宥宁 知夏 若岚 星澜 景初 予安 映雪 承远]).map(&:join)
              else
        %w[Isadora Jocelyn Lenora Miranda Nerissa Ophelia Penelope Rosalind].product(%w[Sutton Fletcher Chapman Barrett Sullivan Bishop Spencer Hamilton]).map { |first, last| "#{first} #{last}" }
              end
      split.to_s == "development" ? names.first(32) : names.last(32)
    end
  end
end
