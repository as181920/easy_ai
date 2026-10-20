module DecisionScaling
  module_function

  def update_ledger(root)
    patterns = %w[preflight/*/training.jsonl arms/*/choice/training.jsonl supervised/*/choice/training.jsonl]
    patterns.flat_map { |pattern| Dir[File.join(root, pattern)] }.sort.to_h do |path|
      [path.delete_prefix(root + "/"), File.foreach(path).count { |line| !line.strip.empty? }]
    end
  end

  def enforce_budget!(root, remaining)
    raise ArgumentError, "Invalid remaining budget" unless remaining.is_a?(Integer) && remaining >= 0
    spent = update_ledger(root).values.sum
    raise "Scaling update ceiling exceeded: #{spent}+#{remaining}>#{UPDATE_LIMIT}" if spent + remaining > UPDATE_LIMIT
    spent
  end
end
