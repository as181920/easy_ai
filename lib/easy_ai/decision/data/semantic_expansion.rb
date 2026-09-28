module EasyAI
  module Decision
    module Data
      # Reviewed task/candidate paraphrases. Never rewrite facts or infer new gold labels.
      module SemanticExpansion
        VERSION = 1
        NLI_PREFIX = "根据上下文，以下陈述是否成立："
        OPTIONS = {
          "OCNLI" => { "yes" => "上下文支持这一陈述", "no" => "上下文与这一陈述矛盾", "unknown" => "仅凭上下文无法确定" },
          "DuReader-YesNo" => { "yes" => "肯定回答", "no" => "否定回答", "depends" => "需要视情况而定" },
          "BoolQ" => { "yes" => "Yes, according to the passage", "no" => "No, according to the passage" }
        }.freeze

        def self.call(row)
          labels = OPTIONS.fetch(row.source)
          raise ArgumentError, "Unexpected candidates for #{row.source}" unless row.options.map { |option| option.fetch("id") }.sort == labels.keys.sort
          question = case row.source
                     when "OCNLI"
            raise ArgumentError, "Unknown NLI question format" unless row.question.start_with?(NLI_PREFIX)
            "请判断下面的陈述与上下文的关系：#{row.question.delete_prefix(NLI_PREFIX)}"
                     when "DuReader-YesNo"
            "针对问题“#{row.question}”，以上回答的态度是什么？"
                     when "BoolQ"
            "Based on the passage, #{row.question}"
                     end
          row.to_h.merge("id" => "expression-v#{VERSION}:#{row.id}", "question" => question,
            "options" => row.options.map { |option| option.merge("text" => labels.fetch(option.fetch("id"))) },
            "augmentation" => { "version" => VERSION, "parent_id" => row.id, "kind" => "task_and_candidate_paraphrase" })
        end
      end
    end
  end
end
