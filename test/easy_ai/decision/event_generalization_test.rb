require "test_helper"
require_relative "../../../benchmarks/decision/event_generalization"

# Joint assertions verify cross-row alignment and exact checkpoint boundaries.
# rubocop:disable Minitest/MultipleAssertions
class EventGeneralizationTest < Minitest::Test
  V08 = DecisionEventGeneralization

  def rows(axis = :event, language = "en-US", group = "test:train:0")
    V08::Corpus.world_rows(["Carla Hill", "Dora Pine"], %w[work shopping], language: language, group: group, axis: axis)
  end

  def test_explicit_bilingual_renderings_gold_and_complete_pairs
    event = rows

    assert_equal 32, event.size
    assert_equal "Carla Hill did not attend the meeting. Carla Hill did not buy a coat.", event.first.fetch("state")
    assert_equal "Judge this claim against the record: Carla Hill attended the meeting.", event.first.fetch("question")
    assert_equal "no", V08::Corpus.gold!(event.first)
    mixed = event.find { |row| row.dig("world", "mixed") && row.dig("world", "query") == 1 && row.dig("world", "assertion") }

    assert_equal "yes", V08::Corpus.gold!(mixed)
    chinese = V08::Corpus.world_rows(%w[甲同学 乙同学], %w[cat wall], language: :"zh-CN", group: "zh", axis: :event).first

    assert_equal "甲同学没有喂猫。甲同学没有粉刷墙壁。", chinese.fetch("state")
    assert_equal "根据记录判断陈述：甲同学喂了猫。", chinese.fetch("question")
    assert_raises(RuntimeError) { V08::Corpus.gold!(chinese.merge("target" => "yes")) }
    metric = DecisionJudgment.metrics(event + rows(:actor, "en-US", "actor"), (event + rows(:actor, "en-US", "actor")).map { |row| row.fetch("target") == "yes" ? [1.0, -1.0] : [-1.0, 1.0] })

    assert_equal 8, metric.dig("by_language", "en-US", "event_binding_groups")
    assert_equal 8, metric.dig("by_language", "en-US", "actor_binding_groups")
    assert_in_delta 1, metric.dig("by_language", "en-US", "balanced")
    assert_equal 1, (event.map { |row| V08::Catalog.family(row) }).uniq.size
    assert_equal V08::Catalog.material("A: X. B: Y."), V08::Catalog.material("a x。b y")
  end

  def test_event_loss_hand_values_gradients_candidate_permutation_and_axis_isolation
    raw = rows.select { |row| row.dig("world", "mixed") }.first(2) + rows(:actor).select { |row| row.dig("world", "mixed") }.first(2)
    examples = raw.map { |row| EasyAI::Decision::Data::Example.new(row) }
    scores = Torch.zeros([4, 2], requires_grad: true)
    loss = V08::PairLoss.call(scores, examples, axis: "event")

    assert_in_delta Math.log(1 + Math.exp(1)), loss.item, 1e-6
    loss.backward
    gradients = scores.grad.to_a

    examples.first(2).each_with_index { |row, index| assert_operator gradients[index][row.target_index], :<, 0 }
    assert_equal [[0.0, 0.0], [0.0, 0.0]], gradients.last(2)
    correct = Torch.tensor(examples.map { |row| row.options.map { |option| option.fetch("id") == row.target ? 1.0 : -1.0 } })
    reversed = examples.map { |row| EasyAI::Decision::Data::Example.new(row.to_h.merge("options" => row.options.reverse)) }

    assert_in_delta Math.log(1 + Math.exp(-3)), V08::PairLoss.call(correct, examples).item, 1e-6
    assert_in_delta V08::PairLoss.call(correct, examples).item, V08::PairLoss.call(Torch.tensor(correct.to_a.map(&:reverse)), reversed).item, 1e-6
    assert_raises(ArgumentError) { V08::PairLoss.call(scores, examples, axis: :other) }
    assert_raises(ArgumentError) { V08::PairLoss.call(Torch.zeros([1, 2]), examples.first(1)) }
  end

  def test_coverage_changes_exactly_one_unit_per_bucket_and_preserves_truth_order_and_catalog_separation
    original = rows(:actor) + rows(:event, "en-US", "test:1")
    unit = original.first(8).map { |row| EasyAI::Decision::Data::Example.new(row) }
    augmenter = V08::Augmentation.new(rows: original)
    changed = augmenter.call(unit, rng: Random.new(5), coverage: true)

    assert_equal original.first(4).map { |row| row.fetch("state") }, changed.first(4).map { |row| row.fetch("state") }
    assert_equal unit.map(&:target), changed.map { |row| V08::Corpus.gold!(row) }
    assert changed.last(4).all? { |row| V08::Catalog::POOLS.fetch("train").include?(row.dig("world", "query_event")) }
    assert changed.last(4).all? { |row| V08::Catalog::FORMS.fetch("train").include?(row.dig("world", "form")) }
    assert_equal changed, augmenter.call(unit, rng: Random.new(5), coverage: true)
    assert_equal unit.map(&:state), augmenter.call(unit, rng: Random.new(5), coverage: false).map { |row| row.fetch("state") }
    assert_equal 16, augmenter.call(unit + unit, rng: Random.new(5)).size
    assert_raises(ArgumentError) { augmenter.call(unit.first(3), rng: Random.new(1)) }
    V08::Catalog::POOLS.values.combination(2) { |a, b| assert_empty a & b }
    V08::Catalog::FORMS.values.combination(2) { |a, b| assert_empty a & b }
    V08::Catalog::LANGUAGES.each { |language| assert_empty V08::Catalog.novel_names(language, :development) & V08::Catalog.novel_names(language, :lockbox) }
    reserved = changed.last(4).map { |row| V08::Catalog.material(row.fetch("state")) }
    other = V08::Augmentation.new(rows: original, forbidden: reserved).call(unit, rng: Random.new(5))

    assert_empty reserved & other.map { |row| V08::Catalog.material(row.fetch("state")) }
  end

  def test_disjoint_world_allocation_and_catalog_axes
    dev = V08::Corpus.panels(:development, worlds: 4, excluded_families: Set.new, historical_materials: Set.new)
    families = dev.values.flatten.map { |row| V08::Catalog.family(row) }.to_set
    final = V08::Corpus.panels(:lockbox, worlds: 4, excluded_families: families, historical_materials: Set.new)

    assert_equal 5, dev.size
    assert_equal 256, dev.fetch("events").size
    assert_empty families & final.values.flatten.map { |row| V08::Catalog.family(row) }.to_set
    assert dev.fetch("events").all? { |row| V08::Catalog::POOLS.fetch("development").include?(row.dig("world", "query_event")) }
    assert final.fetch("expressions").all? { |row| V08::Catalog::FORMS.fetch("lockbox").include?(row.dig("world", "form")) }
  end

  def test_combined_resume_matches_and_reference_matches_v07_numerically
    original = %w[en-US zh-CN].flat_map do |language|
      2.times.flat_map { |index| V08::Corpus.world_rows(language == "zh-CN" ? %w[甲同学 乙同学] : ["Carla Hill", "Dora Pine"], %w[work shopping], language: language, group: "unit:#{index}", axis: index.even? ? :actor : :event) }
    end
    Dir.mktmpdir do |directory|
      dataset = write_dataset(File.join(directory, "train.jsonl"), original)
      config = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 }, training: { choice_microbatch: 16, gradient_accumulation: 2, steps: 2, track_coverage: true })
      a = EasyAI::Decision::ChoiceModel.new(config)
      b = EasyAI::Decision::ChoiceModel.new(config)
      b.load_state_dict(a.state_dict)
      old = DecisionBindingGeneralization::Trainer.new(model: a, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(directory, "old"), variant: :combined)
      reference = V08::Trainer.new(model: b, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(directory, "reference"), variant: "reference")
      old.train
      reference.train

      assert a.state_dict.all? { |key, tensor| (tensor - b.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal old.state.fetch("coverage"), reference.state.fetch("coverage")
      combined = V08::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(directory, "combined"), variant: :combined)
      combined.train(steps: 1)
      first_checkpoint = combined.last_checkpoint
      resumed = V08::Trainer.resume(combined.last_checkpoint, dataset: dataset, output: File.join(directory, "resumed"), variant: "combined")
      loaded = EasyAI::Decision::Checkpoint.load(first_checkpoint)
      migration = { "previous_signature" => loaded.fetch(:metadata).dig("training", "event_generalization"),
        "new_forbidden_sha256" => Digest::SHA256.hexdigest(JSON.generate(["unused"])), "rendered_state_overlap" => 0 }
      migrated = V08::Trainer.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: File.join(directory, "migrated"),
        variant: :combined, forbidden_states: ["unused"], restored: loaded.fetch(:metadata), migration: migration)
      combined.train
      resumed.train
      migrated.train

      assert combined.model.state_dict.all? { |key, tensor| (tensor - resumed.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert combined.model.state_dict.all? { |key, tensor| (tensor - migrated.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal combined.state.fetch("rendering_coverage"), resumed.state.fetch("rendering_coverage")
      assert_equal combined.state.fetch("coverage"), migrated.state.fetch("coverage")
      assert_raises(ArgumentError) do
        V08::Trainer.new(model: loaded.fetch(:model), tokenizer: loaded.fetch(:tokenizer), dataset: dataset, output: File.join(directory, "unsafe-migration"),
          variant: :combined, forbidden_states: ["unused"], restored: loaded.fetch(:metadata), migration: migration.merge("rendered_state_overlap" => 1))
      end
      assert_raises(ArgumentError) { V08::Trainer.resume(combined.last_checkpoint, dataset: dataset, output: File.join(directory, "invalid"), variant: :coverage) }
      assert_raises(ArgumentError) { V08::Trainer.resume(combined.last_checkpoint, dataset: dataset, output: File.join(directory, "invalid2"), variant: :combined, forbidden_states: ["changed"]) }
    end
  end

  def test_integrity_rejects_rendering_and_pair_tampering
    original = rows

    assert V08.audit_rows!(original)
    altered = original.map(&:dup)
    altered[0] = altered.first.merge("question" => "This is a different claim.")

    assert_raises(RuntimeError) { V08.audit_rows!(altered) }
    assert_raises(RuntimeError) { V08.audit_rows!(original.drop(1)) }
    guesses = original.map { |row| row.fetch("target") }
    metric = V08.independent_pairs(original, guesses, axis: :event)

    assert_equal 8, metric.dig("en-US", "groups")
    assert_in_delta 1, metric.dig("en-US", "all_correct")
    blind = original.map { "yes" }

    assert_in_delta 0, V08.independent_pairs(original, blind, axis: :event).dig("en-US", "all_correct")
    assert_in_delta 0.5, V08.independent_pairs(original, blind, axis: :event).dig("en-US", "row_accuracy")
  end

  def test_direction_requires_event_gain_balanced_retention_and_actor_retention
    make = lambda do |event, actor, balanced|
      metric = { "by_language" => V08::Catalog::LANGUAGES.to_h { |language| [language, { "event_binding_all_correct" => event, "actor_binding_all_correct" => actor, "balanced" => balanced }] } }
      { "development" => V08::Catalog::VIEWS.to_h { |view| [view, Marshal.load(Marshal.dump(metric))] }, "regression" => { "canonical" => metric } }
    end
    reference = make.call(0.25, 0.70, 0.85)

    assert V08.direction?(make.call(0.30, 0.65, 0.82), reference)
    refute V08.direction?(make.call(0.29, 0.70, 0.85), reference)
    refute V08.direction?(make.call(0.30, 0.64, 0.85), reference)
    candidate = make.call(0.30, 0.70, 0.85)
    candidate["development"]["expressions"]["by_language"]["zh-CN"]["balanced"] = 0.81

    refute V08.direction?(candidate, reference)
  end

  def test_sealed_selection_resolves_variant_files_and_lockbox_predictions_are_guarded
    Dir.mktmpdir do |root|
      hashes = V08::VARIANTS.to_h do |variant|
        path = File.join(root, "arms", "#{variant}-1337", "result.json")
        V08.seal(path, { "variant" => variant })
        [variant, V08.sha(path)]
      end
      value = { "improving" => true, "winner" => "combined", "results_sha256" => hashes }
      V08.seal(File.join(root, "selection.json"), value)

      assert_equal value, V08.selection(root)
      assert_raises(RuntimeError) { V08.seal(File.join(root, "selection.json"), value) }
      File.write(File.join(root, "arms/reference-1337/result.json"), "changed")

      assert_raises(RuntimeError) { V08.selection(root) }
      error = assert_raises(RuntimeError) { V08.evaluate_views(root, nil, nil, ["lockbox-events"], "test") }

      assert_equal "Lockbox remains closed", error.message
    end
  end

  def test_reserved_world_is_excluded_under_every_training_template_and_record_order
    original = rows
    held = original.map { |row| V08::Catalog.render(row, form: 3) }
    exclusions = V08.reserved_materials(held)

    refute_includes held.map { |row| V08::Catalog.material(row.fetch("state")) }, V08::Catalog.material(original.first.fetch("state"))
    assert_includes exclusions, V08::Catalog.material(original.first.fetch("state"))
    (0..6).each do |form|
      original.each { |row| assert_includes exclusions, V08::Catalog.material(V08::Catalog.render(row, form: form).fetch("state")) }
    end

    assert_equal 1, held.map { |row| V08::Catalog.family(row) }.uniq.size
  end

  def test_paired_template_diagnostic_preserves_facts_targets_and_family
    %w[en-US zh-CN].each do |language|
      [:actor, :event].each do |axis|
        before = rows(axis, language).map { |row| V08::Catalog.render(row, form: axis == :actor ? 5 : 6) }
        after = before.map { |row| V08.crossed_row(row, split: :lockbox) }

        assert_equal before.map { |row| row.fetch("target") }, after.map { |row| V08::Corpus.gold!(row) }
        assert_equal before.map { |row| V08::Catalog.family(row) }, after.map { |row| V08::Catalog.family(row) }
        assert_equal [axis == :actor ? 6 : 5], after.map { |row| row.dig("world", "form") }.uniq
        refute_equal before.first.fetch("state"), after.first.fetch("state")
        assert_equal before, after.map { |row| V08.crossed_row(row) }
      end
    end
  end
end
