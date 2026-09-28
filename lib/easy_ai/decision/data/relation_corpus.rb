require "digest"
require "json"
require "fileutils"

module EasyAI
  module Decision
    module Data
      # A closed, explicitly boolean world. Negating a question is not used to
      # label uncertain/modal language: those phenomena need a separate task.
      class RelationCorpus
        VERSION = 2
        ACTORS = [
          ["小林", "Alice"], ["小周", "Bob"], ["小陈", "Carol"], ["小王", "David"],
          ["小李", "Emma"], ["小张", "Frank"], ["小吴", "Grace"], ["小赵", "Henry"]
        ].freeze
        ACTIONS = [
          ["买了票", "没有买票", "bought a ticket", "did not buy a ticket"],
          ["打开了门", "没有打开门", "opened the door", "did not open the door"],
          ["点亮了灯", "没有点亮灯", "turned on the light", "did not turn on the light"],
          ["关闭了窗户", "没有关闭窗户", "closed the window", "did not close the window"],
          ["带了雨伞", "没有带雨伞", "brought an umbrella", "did not bring an umbrella"],
          ["完成了作业", "没有完成作业", "finished the homework", "did not finish the homework"]
        ].freeze
        SPLITS = %w[train validation calibration test test-familiar].freeze

        def initialize(seed: 1337)
          @seed = seed
        end

        def write(output:, vocab_size: 400)
          raise ArgumentError, "Output exists: #{output}" if File.exist?(output)
          families = (0...ACTORS.size).to_a.combination(2).flat_map do |actors|
            ACTIONS.each_index.map { |action| [*actors, action] }
          end.sort_by { |family| digest([@seed, family]) }
          rows = SPLITS.to_h { |split| [split, []] }
          families.each_with_index do |family, index|
            split, styles = case index
                            when 0...108 then ["train", [0, 1]]
                            when 108...128 then ["validation", [2]]
                            when 128...148 then ["calibration", [2]]
                            else ["test", [3]]
                            end
            styles.each { |style| rows[split].concat(examples(family, style: style)) }
            rows["test-familiar"].concat(examples(family, style: 0)) if split == "test"
          end
          FileUtils.mkdir_p(output)
          rows.each { |split, items| write_rows(File.join(output, "#{split}.jsonl"), items) }
          write_rows(File.join(output, "sanity.jsonl"), examples(families.first, style: 0))
          tokenizer = Tokenizers::NativeBpe.new
          tokenizer.train(rows.fetch("train").flat_map { |row| Example.new(row).texts }, vocab_size: vocab_size)
          tokenizer.save(File.join(output, "tokenizer.json"))
          Dataset.assert_disjoint!(*%w[train validation calibration test].map { |split| Dataset.new(File.join(output, "#{split}.jsonl")) })
          manifest = { "version" => VERSION, "seed" => @seed, "families" => families.size,
            "rows" => rows.transform_values(&:size), "sanity_rows" => 64,
            "tokenizer_fingerprint" => tokenizer.fingerprint, "tokenizer_vocab_size" => tokenizer.vocab_size,
            "files_sha256" => Dir.glob(File.join(output, "*.jsonl")).to_h { |path| [File.basename(path), Digest::SHA256.file(path).hexdigest] },
            "split_policy" => "All translations, fact flips, question flips and order variants of an actor-pair/action family stay together. Train styles 0/1, validation/calibration style 2, test style 3. test-familiar shares test families; sanity is a training subset, never a generalization test.",
            "scope" => "Synthetic explicit facts only: binding, negated assertions, irrelevant negation, order and candidate permutation. No claim about modal/conditional language or general semantics." }
          File.write(File.join(output, "manifest.json"), JSON.pretty_generate(manifest))
          manifest
        end

        def examples(family, style:)
          left, right, action = family
          group = "relation:#{left}:#{right}:#{action}"
          [false, true].repeated_permutation(2).flat_map do |facts|
            [0, 1].product([false, true], ["zh-CN", "en-US"], [0, 1]).map do |subject, assertion, language, order|
              query = clause(family[subject], action, assertion, language)
              clauses = [left, right].each_with_index.map { |actor, i| clause(actor, action, facts[i], language) }
              clauses.reverse! if order == 1
              state, question = render(clauses, query, language, style)
              identity = [group, facts, subject, assertion, language, order, style]
              options = language == "zh-CN" ? { "yes" => "成立", "no" => "不成立" } : { "yes" => "true", "no" => "false" }
              checks = {
                "question_flip" => [group, facts, subject, language, order, style],
                "fact_flip" => [group, facts[1 - subject], subject, assertion, language, order, style],
                "irrelevant_fact" => [group, facts[subject], subject, assertion, language, order, style],
                "order" => [group, facts, subject, assertion, language, style]
              }.transform_values { |key| digest(key) }
              { "id" => digest(identity), "group_id" => group, "source" => "relations", "language" => language,
                "contrast_group" => checks.fetch("question_flip"),
                "state" => state, "question" => question,
                "options" => options.map { |id, text| { "id" => id, "text" => text } }.shuffle(random: Random.new(digest(identity).to_i(16))),
                "target" => facts[subject] == assertion ? "yes" : "no",
                "relation" => { "family" => family, "facts" => facts, "subject" => subject, "assertion" => assertion, "style" => style, "checks" => checks } }
            end
          end
        end

        private

        def clause(actor, action, positive, language)
          offset = language == "zh-CN" ? 0 : 2
          name = ACTORS.fetch(actor).fetch(offset / 2)
          verb = ACTIONS.fetch(action).fetch(offset + (positive ? 0 : 1))
          language == "zh-CN" ? "#{name}#{verb}" : "#{name} #{verb}"
        end

        def render(clauses, query, language, style)
          if language == "zh-CN"
            case style
            when 0 then [clauses.join("，") + "。", "以下说法成立吗：#{query}。"]
            when 1 then ["记录：#{clauses.join('；')}。", "判断这句话是否成立：#{query}。"]
            when 2 then ["#{clauses.first}。另外，#{clauses.last}。", "根据记录判断：#{query}。"]
            when 3 then ["记录中的事实是：#{clauses.join('，并且')}。", "关于这份记录，以下说法是否成立：#{query}。"]
            else raise ArgumentError, "Unknown template style"
            end
          else
            case style
            when 0 then [clauses.join(". ") + ".", "Is this statement true: #{query}?"]
            when 1 then ["Record: #{clauses.join('; ')}.", "Decide whether this claim is true: #{query}."]
            when 2 then ["#{clauses.first}. Also, #{clauses.last}.", "According to the record: #{query}. True or false?"]
            when 3 then ["The recorded facts are: #{clauses.join(', and ')}.", "Does the record support this statement: #{query}?"]
            else raise ArgumentError, "Unknown template style"
            end
          end
        end

        def write_rows(path, rows)
          File.write(path, rows.map { |row| JSON.generate(row) }.join("\n") + "\n")
        end

        def digest(value)
          Digest::SHA256.hexdigest(JSON.generate(value))
        end
      end
    end
  end
end
