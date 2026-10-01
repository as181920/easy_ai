require "digest"
require "json"
require "fileutils"

module EasyAI
  module Decision
    module Data
      # Ground truth comes from explicit boolean records during preparation only.
      # Inference never calls this generator or interprets its world metadata.
      class EvidenceCorpus
        VERSION = 1
        ACTORS = RelationCorpus::ACTORS + [
          ["小杨", "Iris"], ["小徐", "Jack"], ["小孙", "Karen"], ["小郑", "Leo"],
          ["小何", "Maya"], ["小马", "Noah"], ["小郭", "Olivia"], ["小罗", "Peter"]
        ]
        ACTIONS = [
          ["买了票", "没有买票", "bought a ticket", "did not buy a ticket", "购票成功", "没有购票", "purchased a ticket", "did not purchase a ticket"],
          ["打开了门", "没有打开门", "opened the door", "did not open the door", "把门打开了", "未把门打开", "opened up the door", "did not open up the door"],
          ["点亮了灯", "没有点亮灯", "turned on the light", "did not turn on the light", "开了灯", "没有开灯", "switched on the light", "did not switch on the light"],
          ["关闭了窗户", "没有关闭窗户", "closed the window", "did not close the window", "关上了窗户", "没关上窗户", "shut the window", "did not shut the window"],
          ["带了雨伞", "没有带雨伞", "brought an umbrella", "did not bring an umbrella", "携带了雨伞", "未携带雨伞", "took an umbrella along", "did not take an umbrella along"],
          ["完成了作业", "没有完成作业", "finished the homework", "did not finish the homework", "做完了作业", "没做完作业", "completed the homework", "did not complete the homework"],
          ["收到了邮件", "没有收到邮件", "received an email", "did not receive an email", "收到了一封电子邮件", "没收到电子邮件", "got an email", "did not get an email"],
          ["洗了杯子", "没有洗杯子", "washed the cup", "did not wash the cup", "清洗了杯子", "未清洗杯子", "cleaned the cup", "did not clean the cup"],
          ["提交了报告", "没有提交报告", "submitted the report", "did not submit the report", "交了报告", "没交报告", "handed in the report", "did not hand in the report"],
          ["锁好了车", "没有锁车", "locked the car", "did not lock the car", "把车锁上了", "没把车锁上", "locked up the car", "did not lock up the car"],
          ["买了牛奶", "没有买牛奶", "bought milk", "did not buy milk", "购买了牛奶", "未购买牛奶", "purchased milk", "did not purchase milk"],
          ["预订了房间", "没有预订房间", "booked a room", "did not book a room", "订好了房间", "没订房间", "reserved a room", "did not reserve a room"]
        ].freeze
        SIZES = { "train" => 64, "validation" => 16, "calibration" => 16, "test" => 32 }.freeze

        def initialize(seed: 9049)
          @seed = seed
        end

        def write(output:)
          raise ArgumentError, "Output exists: #{output}" if File.exist?(output)
          families = ACTORS.each_index.to_a.combination(2).flat_map { |pair| ACTIONS.each_index.map { |action| [*pair, action] } }
            .sort_by { |family| digest([@seed, family]) }
          FileUtils.mkdir_p(output)
          cursor, chosen = 0, {}
          SIZES.each do |split, size|
            selected = families.slice(cursor, size)
            cursor += size
            chosen[split] = selected
            styles = split == "train" ? [0, 1] : [split == "test" ? 3 : 2]
            write_rows(output, split, selected.flat_map { |family| styles.flat_map { |style| examples(family, style: style) } })
          end
          write_rows(output, "test-familiar", chosen.fetch("test").flat_map { |family| examples(family, style: 0) })
          sanity = chosen.fetch("train").first(2).flat_map { |family| examples(family, style: 0).select { |row| row.dig("world", "distractor") == "none" } }
          write_rows(output, "sanity", sanity)
          Dataset.assert_disjoint!(*SIZES.keys.map { |split| Dataset.new(File.join(output, "#{split}.jsonl")) })
          manifest = { "version" => VERSION, "seed" => @seed, "families" => chosen,
            "files_sha256" => Dir.glob(File.join(output, "*.jsonl")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
            "scope" => "Controlled generated explicit facts, not a natural-language benchmark. No lateness examples, modality, missing evidence or open-world claims.",
            "split_policy" => "Actor-pair/action families, all translations, truth flips, orders and distractors stay together. Train styles 0/1, validation/calibration 2, test 3 with new verb expressions. Familiar test shares test families; sanity is a training subset." }
          File.write(File.join(output, "manifest.json"), JSON.pretty_generate(manifest) + "\n")
          manifest
        end

        def examples(family, style:)
          [false, true].repeated_permutation(2).flat_map do |facts|
            [0, 1].product([false, true], %w[zh-CN en-US], [0, 1], %w[none positive negative]).map do |subject, assertion, language, order, distractor|
              build(family, facts, subject, assertion, language, order, distractor, style)
            end
          end
        end

        private

        def build(family, facts, subject, assertion, language, order, distractor, style)
          left, right, action = family
          actors = [left, right]
          extra = (ACTORS.each_index.to_a - actors).fetch(digest(family).to_i(16) % (ACTORS.size - 2))
          records = actors.each_with_index.map { |actor, index| [actor, action, facts[index]] }
          records << [extra, (action + 1) % ACTIONS.size, distractor == "positive"] unless distractor == "none"
          records.reverse! if order == 1
          state = records.map { |actor, verb, positive| sentence(actor, verb, positive, language, style) }.join("\n")
          query = clause(actors.fetch(subject), action, assertion, language)
          question = questions(language, style, query)
          group = "evidence-v#{VERSION}:#{family.join(':')}"
          identity = [group, facts, subject, assertion, language, order, distractor, style]
          checks = {
            "fact_flip" => [group, facts[1 - subject], subject, assertion, language, order, distractor, style],
            "question_flip" => [group, facts, subject, language, order, distractor, style],
            "subject_switch" => [group, facts, assertion, language, order, distractor, style],
            "order" => [group, facts, subject, assertion, language, distractor, style],
            "irrelevant_fact" => [group, facts, subject, assertion, language, order, style],
            "binding" => [group, facts.uniq.size == 2, assertion, language, order, distractor, style]
          }.transform_values { |value| digest(value) }
          options = options(language, style).shuffle(random: Random.new(digest(identity).to_i(16)))
          { "id" => digest(identity), "group_id" => group, "source" => "evidence", "language" => language,
            "state" => state, "question" => question, "options" => options, "target" => facts.fetch(subject) == assertion ? "yes" : "no",
            "evidence_index" => records.index { |record| record.first == actors.fetch(subject) },
            "world" => { "family" => family, "facts" => facts, "subject" => subject, "assertion" => assertion, "order" => order,
              "style" => style, "distractor" => distractor, "records" => records, "checks" => checks } }
        end

        def clause(actor, action, positive, language, variant = 0)
          offset = language == "zh-CN" ? 0 : 2
          name = ACTORS.fetch(actor).fetch(offset / 2)
          verb = ACTIONS.fetch(action).fetch(variant * 4 + offset + (positive ? 0 : 1))
          language == "zh-CN" ? "#{name}#{verb}" : "#{name} #{verb}"
        end

        def sentence(actor, action, positive, language, style)
          text = clause(actor, action, positive, language, style == 3 ? 1 : 0)
          if language == "zh-CN"
            ["#{text}。", "已知事实：#{text}。", "记录确认，#{text}。", "调查记录显示，#{text}。"].fetch(style)
          else
            ["#{text}.", "We know that #{text}.", "The record confirms that #{text}.", "It is documented that #{text}."].fetch(style)
          end
        end

        def questions(language, style, query)
          if language == "zh-CN"
            ["以下说法成立吗：#{query}。", "根据记录，#{query}，对吗？", "这些事实是否支持：#{query}？", "请判断这项陈述是否正确：#{query}。"].fetch(style)
          else
            ["Is this statement true: #{query}?", "According to the record, is it true that #{query}?",
             "Do these facts support the claim that #{query}?", "Does the documented information establish that #{query}?"].fetch(style)
          end
        end

        def options(language, style)
          texts = if language == "zh-CN"
            [["成立", "不成立"], ["是", "不是"], ["支持", "矛盾"], ["正确", "错误"]].fetch(style)
                  else
            [["true", "false"], ["yes", "no"], ["supported", "contradicted"], ["correct", "incorrect"]].fetch(style)
                  end
          %w[yes no].zip(texts).map { |id, text| { "id" => id, "text" => text } }
        end

        def digest(value)
          Digest::SHA256.hexdigest(JSON.generate(value))
        end

        def write_rows(output, split, rows)
          File.write(File.join(output, "#{split}.jsonl"), rows.map { |row| JSON.generate(row) }.join("\n") + "\n")
        end
      end
    end
  end
end
