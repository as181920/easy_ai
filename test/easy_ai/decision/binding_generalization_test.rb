require "test_helper"
require_relative "../../../benchmarks/decision/binding_generalization"

# Joint assertions check alignment across paired rows and checkpoint boundaries.
# rubocop:disable Minitest/MultipleAssertions
class BindingGeneralizationTest < Minitest::Test
  def actor_rows(language = "en-US")
    corpus = EasyAI::Decision::Data::ScalableJudgmentCorpus
    corpus.new(:train, 0, language, counts: { train: 2 }).rows.select { |row| row.dig("world", "profile") == "binary" && row.dig("world", "mixed") }.first(4)
  end

  def test_pair_loss_hand_values_gradient_signs_permutations_and_duplicate_sampling
    raw = actor_rows.first(2)
    examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    logits = Torch.zeros([2, 2], requires_grad: true)
    loss = EasyAI::Decision::BindingPairLoss.call(logits, examples)

    assert_in_delta Math.log(1 + Math.exp(1)), loss.item, 1e-6
    loss.backward
    gradients = logits.grad.to_a
    examples.each_with_index do |row, index|
      target = row.target_index

      assert_operator gradients[index][target], :<, 0
      assert_operator gradients[index][1 - target], :>, 0
    end
    reversed = examples.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) }
    better = Torch.tensor(examples.map { |row| row.options.map { |option| option.fetch("id") == row.target ? 1.0 : -1.0 } })

    assert_in_delta Math.log(1 + Math.exp(-3)), EasyAI::Decision::BindingPairLoss.call(better, examples).item, 1e-6
    assert_in_delta loss.item, EasyAI::Decision::BindingPairLoss.call(Torch.zeros([2, 2]), reversed).item, 1e-6
    assert_in_delta EasyAI::Decision::BindingPairLoss.call(better, examples).item,
      EasyAI::Decision::BindingPairLoss.call(Torch.tensor(better.to_a.map(&:reverse)), reversed).item, 1e-6
    wrong_targets = examples.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("target" => row.target == "yes" ? "no" : "yes")) }

    assert_in_delta Math.log(1 + Math.exp(5)), EasyAI::Decision::BindingPairLoss.call(better, wrong_targets).item, 1e-6
    assert_in_delta loss.item, EasyAI::Decision::BindingPairLoss.call(Torch.zeros([4, 2]), examples + examples).item, 1e-6
    assert_raises(ArgumentError) { EasyAI::Decision::BindingPairLoss.call(Torch.zeros([1, 2]), examples.first(1)) }
    assert_raises(ArgumentError) { EasyAI::Decision::BindingPairLoss.call(Torch.zeros([2, 2]), [examples.first, examples.first]) }
    homogeneous = examples.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("target" => "yes")) }

    assert_in_delta 0, EasyAI::Decision::BindingPairLoss.call(Torch.zeros([2, 2]), homogeneous).item, 1e-6
  end

  def test_renaming_is_consistent_gold_preserving_stateless_and_rejects_reserved_material
    raw = actor_rows
    examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    actors = raw.to_h { |row| [row.fetch("id"), row.dig("world", "facts").map { |fact| fact.fetch("actor") }.uniq] }
    pool = ["Carla Hill", "Dora Pine"]
    make = ->(forbidden = []) { EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: { "en-US" => pool }, forbidden_states: forbidden) }
    renamed = make.call.call(examples, rng: Random.new(41))
    mapping = examples.first(2).zip(renamed.first(2)).to_h do |before, after|
      original = actors.fetch(before.id).find { |name| before.question.include?(name) }
      replacement = pool.find { |name| after.question.include?(name) }
      [original, replacement]
    end
    checked = raw.zip(renamed).map do |before, after|
      world = Marshal.load(Marshal.dump(before.fetch("world")))
      world.fetch("facts").each { |fact| fact["actor"] = mapping.fetch(fact.fetch("actor")) }
      world["actor"] = mapping.fetch(world.fetch("actor"))
      after.to_h.merge("world" => world)
    end

    assert EasyAI::Decision::Data::QualityAudit.integrity!(checked)
    assert_equal 1, renamed.map(&:state).uniq.size
    assert_equal examples.map(&:target), renamed.map(&:target)
    assert_equal renamed.map(&:to_h), make.call.call(examples, rng: Random.new(41)).map(&:to_h)
    assert_equal 8, make.call.call(examples + examples, rng: Random.new(41)).size
    refute_equal examples.first.state, renamed.first.state
    forbidden = renamed.map { |row| Digest::SHA256.hexdigest(row.state) }.uniq
    other = make.call(forbidden).call(examples, rng: Random.new(41))

    refute_includes forbidden, Digest::SHA256.hexdigest(other.first.state)
    assert_raises(ArgumentError) { make.call.call(examples.first(3), rng: Random.new(1)) }
    all_states = pool.permutation.map do |names|
      replacements = actors.values.first.zip(names).to_h
      Digest::SHA256.hexdigest(examples.first.state.gsub(Regexp.union(replacements.keys)) { |match| replacements.fetch(match) })
    end

    assert_raises(ArgumentError) { make.call(all_states).call(examples, rng: Random.new(1)) }
    held_hashes = DecisionBindingGeneralization.held_state_hashes(raw.first)
    reversed_state = raw.first.fetch("state").split(".").map(&:strip).reverse.join(". ") + "."

    assert_includes held_hashes, Digest::SHA256.hexdigest(raw.first.fetch("state"))
    assert_includes held_hashes, Digest::SHA256.hexdigest(reversed_state)
    original_actors = actors.values.first
    swapper = EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: { "en-US" => original_actors.reverse })
    swapped = swapper.call(examples, rng: Random.new(0))

    assert_includes swapped.first.question, original_actors.last
    assert_includes swapped.fetch(1).question, original_actors.first
    original_actors.each { |actor| assert_equal 1, swapped.first.state.scan(Regexp.new(Regexp.escape(actor))).size }
  end

  def test_ce_path_matches_previous_trainer_and_combined_resume_is_exact
    raw = 2.times.flat_map do |index|
      %w[en-US zh-CN].flat_map { |language| EasyAI::Decision::Data::ScalableJudgmentCorpus.new(:train, index, language, counts: { train: 2 }).rows }
    end
    raw = DecisionScaling.binary_rows(raw)
    Dir.mktmpdir do |directory|
      dataset = write_dataset(File.join(directory, "train.jsonl"), raw)
      config = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 }, training: { choice_microbatch: 4, gradient_accumulation: 8, steps: 2, track_coverage: true })
      Torch.manual_seed(3)
      model = EasyAI::Decision::ChoiceModel.new(config)
      copy = EasyAI::Decision::ChoiceModel.new(config)
      copy.load_state_dict(model.state_dict)
      previous = DecisionScaling::QuadTrainer.new(model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(directory, "old"))
      current = DecisionBindingGeneralization::Trainer.new(model: copy, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(directory, "new"), variant: :ce)
      previous.train
      current.train

      assert previous.model.state_dict.all? { |key, tensor| (tensor - current.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal previous.state.fetch("coverage"), current.state.fetch("coverage")
      combined = DecisionBindingGeneralization::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new,
        dataset: dataset, output: File.join(directory, "combined"), variant: :combined)
      combined.train(steps: 1)
      resumed = DecisionBindingGeneralization::Trainer.resume(combined.last_checkpoint, dataset: dataset, output: File.join(directory, "resumed"), variant: "combined")
      combined.train
      resumed.train

      assert combined.model.state_dict.all? { |key, tensor| (tensor - resumed.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal combined.state.fetch("coverage"), resumed.state.fetch("coverage")
      assert_raises(ArgumentError) { DecisionBindingGeneralization::Trainer.resume(combined.last_checkpoint, dataset: dataset, output: File.join(directory, "invalid"), variant: :rename) }
      assert_raises(ArgumentError) { DecisionBindingGeneralization::Trainer.resume(combined.last_checkpoint, dataset: dataset, output: File.join(directory, "invalid2"), variant: :combined, forbidden_states: ["different"]) }
    end
  end

  def test_chinese_renaming_preserves_rendered_gold_and_distinct_actor_spans
    raw = actor_rows(:"zh-CN")
    examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    actors = raw.to_h { |row| [row.fetch("id"), row.dig("world", "facts").map { |fact| fact.fetch("actor") }.uniq] }
    pool = %w[甲同学 乙同学]
    renamed = EasyAI::Decision::EntityRenaming.new(actors_by_id: actors, pools: { "zh-CN" => pool }).call(examples, rng: Random.new(0))
    mapping = actors.values.first.zip(pool).to_h
    checked = raw.zip(renamed).map do |before, after|
      world = Marshal.load(Marshal.dump(before.fetch("world")))
      world.fetch("facts").each { |fact| fact["actor"] = mapping.fetch(fact.fetch("actor")) }
      world["actor"] = mapping.fetch(world.fetch("actor"))
      after.to_h.merge("world" => world)
    end

    assert EasyAI::Decision::Data::QualityAudit.integrity!(checked)
    assert_equal examples.map(&:target), renamed.map(&:target)
    pool.each { |actor| assert_equal 1, renamed.first.state.scan(Regexp.new(actor)).size }
    assert_includes renamed.first.question, pool.first
    assert_includes renamed.fetch(1).question, pool.last
  end

  def test_direction_requires_group_gain_without_hidden_language_regression
    metric = ->(accuracy, groups) { { "development" => %w[canonical seen_combinations novel_actor].to_h do |name|
      [name, { "by_language" => %w[en-US zh-CN].to_h { |language| [language, { "balanced" => accuracy, "actor_binding_all_correct" => groups }] } }]
    end } }
    rules = { "worst_actor_gain" => 0.05, "maximum_balanced_drop" => 0.03 }
    reference = metric.call(0.85, 0.25)
    candidate = metric.call(0.82, 0.30)

    assert DecisionBindingGeneralization.direction?(candidate, reference, rules)
    candidate["development"]["novel_actor"]["by_language"]["zh-CN"]["balanced"] = 0.81

    refute DecisionBindingGeneralization.direction?(candidate, reference, rules)
    refute DecisionBindingGeneralization.direction?(metric.call(0.90, 0.29), reference, rules)
  end

  def test_selection_checks_variant_result_files_and_detects_tampering
    Dir.mktmpdir do |root|
      hashes = DecisionBindingGeneralization::VARIANTS.to_h do |variant|
        path = File.join(root, "arms", "#{variant}-1337", "result-6000.json")
        FileUtils.mkdir_p(File.dirname(path))
        File.write(path, JSON.generate("variant" => variant))
        [variant, DecisionBindingGeneralization.sha(path)]
      end
      path = File.join(root, "selection.json")
      value = { "improving" => true, "winner" => "combined", "results_sha256" => hashes }
      DecisionBindingGeneralization.write(path, value)
      DecisionBindingGeneralization.write(path + ".sha256.json", { "sha256" => DecisionBindingGeneralization.sha(path) })

      assert_equal value, DecisionBindingGeneralization.decision(root)
      File.write(File.join(root, "arms/ce-1337/result-6000.json"), "changed")

      assert_raises(RuntimeError) { DecisionBindingGeneralization.decision(root) }
      File.write(path, JSON.generate(value.merge("winner" => "ce")))

      assert_raises(RuntimeError) { DecisionBindingGeneralization.decision(root) }
    end
  end
end
