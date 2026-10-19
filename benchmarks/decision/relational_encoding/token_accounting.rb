module DecisionRelational
  module_function

  def batch_token_counts(lengths, mode)
    mode = mode.to_s
    raise ArgumentError, "Unknown encoding mode" unless %w[separate joint].include?(mode)
    size = lengths.size
    candidates = lengths.map { |row| row.fetch("options").size }.max
    if mode == "separate"
      valid = lengths.sum { |row| row.fetch("state") + row.fetch("options").sum }
      padded = size * lengths.map { |row| row.fetch("state") }.max + size * candidates * lengths.flat_map { |row| row.fetch("options") }.max
    else
      sequences = lengths.flat_map { |row| row.fetch("options").map { |length| row.fetch("state") + length - 1 } }
      valid = sequences.sum
      padded = size * candidates * sequences.max
    end
    { "valid" => valid, "padded" => padded }
  end

  def account_fit_tokens(root)
    verify(root)
    raw = rows(File.join(root, "data/fit.jsonl"))
    languages = raw.each_index.group_by { |index| raw[index].fetch("language") }
    result = Dir.glob(File.join(root, "fit-*/result.json")).sort.to_h do |path|
      fit = JSON.parse(File.read(path))
      metadata = JSON.parse(File.read(File.join(fit.fetch("checkpoint"), "metadata.json")))
      config = EasyAI::Decision::Config.new(metadata.fetch("config"))
      flags = %w[balance_sources balance_labels paired_sampling resample_negatives]
      raise "Accounting supports only the frozen uniform fitting sampler" if flags.any? { |key| config[:training].fetch(key) }
      tokenizer = EasyAI::Tokenizers::Registry.load(File.join(fit.fetch("checkpoint"), "tokenizer.json"))
      collator = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config)
      lengths = raw.map { |row| { "state" => collator.state_tokens(row.fetch("state")).size,
        "options" => row.fetch("options").map { |option| collator.option_tokens(row.fetch("question"), option.fetch("text")).size } } }
      state = metadata.fetch("training")
      micro = state.fetch("microbatch")
      raise "Accounting needs a constant microbatch" unless fit.fetch("measurements").all? { |row| row.fetch("microbatch") == micro }
      totals = { "valid" => 0, "padded" => 0 }
      visits = Array.new(raw.size, 0)
      fit.fetch("step").times do |step|
        rng = Random.new(config[:training].fetch("seed") + step * 1009)
        indexes = Array.new(micro * state.fetch("gradient_accumulation")) { languages.fetch(languages.keys.sort.sample(random: rng)).sample(random: rng) }
        indexes.each { |index| visits[index] += 1 }
        indexes.each_slice(micro) do |chunk|
          counts = batch_token_counts(chunk.map { |index| lengths[index] }, config[:model].fetch("encoding_mode"))
          totals.each_key { |key| totals[key] += counts.fetch(key) }
        end
      end
      unless visits == state.dig("coverage", "row_visits") && totals.fetch("valid") == state.dig("coverage", "input_tokens")
        raise "Sampling/token replay differs from committed training evidence"
      end
      [File.basename(File.dirname(path)), totals.merge("scope" => "Exact replay of committed uniform fitting batches, checked against actual row visits and valid-token counters. Padded encoder input counts do not imply equal attention FLOPs.")]
    end
    write(File.join(root, "fit-token-accounting.json"), result)
    puts JSON.pretty_generate(result)
  end
end
