module DecisionRelational
  UPDATE_LIMIT = 20_000
  module_function

  def update_ledger(root)
    patterns = [File.join(root, "fit-*/choice/training.jsonl"), File.join(root, "pilot*/seed-*/choice/training.jsonl")]
    patterns.flat_map { |pattern| Dir.glob(pattern) }.sort.to_h do |path|
      # Count successful updates including replayed uncommitted trace entries.
      [path.delete_prefix(root + "/"), File.foreach(path).count { |line| !line.strip.empty? }]
    end
  end

  def enforce_update_budget!(root, remaining)
    raise ArgumentError, "Invalid remaining budget" unless remaining.is_a?(Integer) && remaining >= 0
    spent = update_ledger(root).values.sum
    raise "Extended update ceiling would be exceeded: #{spent}+#{remaining}>#{UPDATE_LIMIT}" if spent + remaining > UPDATE_LIMIT
    spent
  end
end
