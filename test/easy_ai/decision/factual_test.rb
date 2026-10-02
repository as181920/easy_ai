require "test_helper"

class FactualTest < Minitest::Test
  def contrast_rows
    EasyAI::Decision::Data::FactualContrasts.new("work", 0, "en-US").rows
  end

  def test_margin_respects_candidate_ids_and_keeps_unknown_unconstrained
    rows = contrast_rows.first(2).map { |row| EasyAI::Decision::Data::Example.new(row) }
    unknown = EasyAI::Decision::Data::Example.new(contrast_rows.last)
    logits = Torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).detach.requires_grad!(true)
    loss = EasyAI::Decision::PairMarginLoss.call(logits, rows + [unknown])
    loss.backward

    assert_in_delta 2.0 / 3, loss.item, 1e-6
    assert_tensor_close Torch.tensor([[-1.0 / 3, 1.0 / 3, 0.0], [1.0 / 3, -1.0 / 3, 0.0], [0.0, 0.0, 0.0]]), logits.grad
    reversed = rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) }

    assert_in_delta loss.item, EasyAI::Decision::PairMarginLoss.call(logits, reversed + [unknown]).item, 1e-6
  end

  def test_margin_is_zero_after_both_signed_margins_are_satisfied
    rows = contrast_rows.first(2).map { |row| EasyAI::Decision::Data::Example.new(row) }

    assert_in_delta 0.0, EasyAI::Decision::PairMarginLoss.call(Torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), rows).item
  end

  def test_margin_is_invariant_to_candidate_permutation_with_unequal_logits
    rows = contrast_rows.first(2).map { |row| EasyAI::Decision::Data::Example.new(row) }
    logits = Torch.tensor([[0.8, 0.3, 0.1], [0.5, 0.6, 0.2]])
    reversed = rows.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) }
    reversed_logits = Torch.tensor([[0.1, 0.3, 0.8], [0.2, 0.6, 0.5]])

    assert_in_delta 0.7, EasyAI::Decision::PairMarginLoss.call(logits, rows).item, 1e-6
    assert_in_delta EasyAI::Decision::PairMarginLoss.call(logits, rows).item,
      EasyAI::Decision::PairMarginLoss.call(reversed_logits, reversed).item, 1e-6
  end

  def test_generated_gold_comes_from_the_recorded_fact_and_assertion
    %w[en-US zh-CN].each do |language|
      EasyAI::Decision::Data::FactualContrasts.new("transport", 9, language).rows.each do |row|
        world = row.fetch("world")
        facts, actor = world.values_at("facts", "queried_actor")
        expected = facts.key?(actor) ? (facts.fetch(actor) == world.fetch("assertion") ? "yes" : "no") : "unknown"

        assert_equal expected, row.fetch("target")
        EasyAI::Decision::Data::Example.new(row)
      end
    end
  end

  def test_matched_sampler_keeps_pairs_and_has_exact_five_step_mixture
    rows = %w[en-US zh-CN].flat_map do |language|
      raw = EasyAI::Decision::Data::FactualContrasts.new("work", 1, language).rows
      raw + [raw.last.merge("id" => "routing:#{language}", "source" => "MASSIVE-Scenario")]
    end.map { |row| EasyAI::Decision::Data::Example.new(row) }
    first = EasyAI::Decision::Data::FactualSampler.new(rows)
    second = EasyAI::Decision::Data::FactualSampler.new(rows)
    totals = Hash.new(0)
    5.times do |step|
      indexes = first.sample(32, rng: Random.new(100 + step), step: step)

      assert_equal indexes, second.sample(32, rng: Random.new(100 + step), step: step)
      sampled = indexes.map { |index| rows[index] }
      sampled.each { |row| totals[row.source] += 1 }
      sampled.select { |row| row.source == "Factual-Contrast" }.each_slice(2) do |pair|
        assert_equal %w[no yes], pair.map(&:target).sort
        assert_equal 1, pair.map { |row| row.contrast_groups.fetch("fact_flip") }.uniq.size
      end
    end

    assert_equal [80, 48, 32], totals.values_at("Factual-Uncertainty", "Factual-Contrast", "MASSIVE-Scenario")
  end

  def test_sampler_rejects_falsely_declared_contrast_pairs
    rows = contrast_rows.map { |row| EasyAI::Decision::Data::Example.new(row.merge("target" => "yes")) }

    assert_raises(ArgumentError) { EasyAI::Decision::Data::FactualSampler.new(rows) }
  end

  def test_binding_questions_ask_about_both_recorded_actors
    rows = contrast_rows.select { |row| [2, 3].include?(row.fetch("world").fetch("variant")) }
    actors = rows.map { |row| row.fetch("world").fetch("queried_actor") }.uniq

    assert_equal 2, actors.size
    assert rows.all? { |row| row.fetch("question").include?(row.fetch("world").fetch("queried_actor")) }
    rows.group_by { |row| row.fetch("world").fetch("variant") }.each_value do |pair|
      assert_equal %w[no yes], pair.map { |row| row.fetch("target") }.sort
    end
    other = EasyAI::Decision::Data::FactualContrasts.new("work", 1, "en-US").rows

    refute_equal actors, other.map { |row| row.fetch("world").fetch("queried_actor") }.uniq
  end

  def test_factual_ce_matches_candidate_ce_and_updates_real_parameters
    Dir.mktmpdir do |directory|
      cfg = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 },
        training: { choice_microbatch: 4, gradient_accumulation: 8, track_coverage: true, steps: 1 })
      raw = %w[en-US zh-CN].flat_map do |language|
        rows = EasyAI::Decision::Data::FactualContrasts.new("work", 1, language).rows
        rows + [rows.last.merge("id" => "routing:#{language}", "source" => "MASSIVE-Scenario")]
      end
      dataset = write_dataset(File.join(directory, "data.jsonl"), raw)
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      model = EasyAI::Decision::ChoiceModel.new(cfg)
      other = EasyAI::Decision::ChoiceModel.new(cfg)
      other.load_state_dict(model.state_dict)
      factual = EasyAI::Decision::FactualTrainer.new(model: model, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "factual"))
      candidate = EasyAI::Decision::CandidateTrainer.new(model: other, tokenizer: tokenizer, dataset: dataset, output: File.join(directory, "candidate"))
      batch = raw.first(4).map { |row| EasyAI::Decision::Data::Example.new(row) }

      assert_in_delta candidate.send(:loss_for, batch, seed: 1, teacher: true).item, factual.send(:loss_for, batch, seed: 1, teacher: true).item, 1e-6
      before = model.encoder.embedding.weight.detach.clone
      factual.train

      assert_operator (model.encoder.embedding.weight - before).abs.max.item, :>, 0
      assert_equal 32, factual.state.fetch("coverage").fetch("row_visits").sum
      restored = EasyAI::Decision::Checkpoint.load(factual.last_checkpoint)

      assert_in_delta 0.0, restored.fetch(:metadata).dig("training", "factual_objective", "pair_margin_weight"), 1e-9
      assert_raises(ArgumentError) do
        EasyAI::Decision::FactualTrainer.new(model: restored.fetch(:model), tokenizer: tokenizer, dataset: dataset,
          output: File.join(directory, "changed"), restored: restored.fetch(:metadata), pair_margin_weight: 0.2)
      end
    end
  end
end
