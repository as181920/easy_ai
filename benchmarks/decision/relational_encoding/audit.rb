module DecisionRelational
  module_function

  def audit(root)
    protocol = verify(root)
    loaded = EasyAI::Decision::Checkpoint.load(PARENT)
    tokenizer = loaded.fetch(:tokenizer)
    raw = rows(File.join(root, "data/fit.jsonl"))
    configs = protocol.fetch("configs").transform_values { |value| EasyAI::Decision::Config.new(value) }
    result = configs.to_h do |arm, config|
      collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config)
      examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
      tokens = examples.map { |row| collator.state_tokens(row.state) }
      collisions = examples.each_index.group_by { |i| [tokens[i], examples[i].question, examples[i].options] }
      raise "Opposite labels collapsed by tokenization" if collisions.values.any? { |indexes| indexes.map { |i| examples[i].target }.uniq.size > 1 }
      pairs = examples.each_index.group_by { |i| [examples[i].language, examples[i].question, tokens[i].sort] }.values
        .select { |indexes| indexes.map { |i| examples[i].target }.uniq.size == 2 }
      representatives = %w[en-US zh-CN].to_h do |language|
        indexes = pairs.find { |items| examples[items.first].language == language }
        raise "Missing order-dependent counterfactual" unless indexes
        selected = indexes.uniq { |i| examples[i].target }.first(2)
        raise "Opposite labels have identical ordered IDs" if tokens[selected.first] == tokens[selected.last]
        [language, selected.map do |i|
          row = examples[i]
          batch = collator.call([row])
          if arm == "joint"
            answer = batch.fetch(:joint_ids)[0][0].to_a.zip(batch.fetch(:joint_answer_mask)[0][0].to_a).filter_map { |id, included| id if included }
            raise "Joint answer span mismatch" unless answer == tokenizer.encode(row.options.first.fetch("text"))
          end
          { "id" => row.id, "state" => row.state, "question" => row.question, "target" => row.target,
            "state_ids" => tokens[i], "state_positions" => (0...tokens[i].size).to_a,
            "joint_ids" => arm == "joint" ? batch.fetch(:joint_ids)[0].to_a : nil,
            "joint_positions" => arm == "joint" ? (0...batch.fetch(:joint_ids).shape.last).to_a : nil,
            "joint_answer_mask" => arm == "joint" ? batch.fetch(:joint_answer_mask)[0].to_a : nil }
        end]
      end
      raise "Unexpected truncation" unless collator.truncated.zero?
      [arm, { "rows" => raw.size, "tokenization_collisions_checked" => collisions.size,
        "order_dependent_sets" => pairs.size, "representatives" => representatives }]
    end
    write(File.join(root, "input-audit.json"), result)
    puts "Verified no opposite-label token collisions, retained actor swaps and joint answer spans in EN/ZH"
  end
end
