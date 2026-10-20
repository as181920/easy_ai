require "test_helper"
require_relative "../../../benchmarks/decision/expression_retention"

# Assertions compare whole semantic units and exact optimizer boundaries.
# rubocop:disable Minitest/MultipleAssertions
class ExpressionRetentionTest < Minitest::Test
  V09 = DecisionExpressionRetention

  def rows(axis = :actor, language = "en-US", group = "fixture:0", form = 0)
    names = language.to_s == "zh-CN" ? %w[甲同学 乙同学] : ["Carla Hill", "Dora Pine"]
    V09::Corpus.world_rows(names, %w[work shopping], axis: axis, language: language, group: group, form: form)
  end

  def training_rows
    %w[en-US zh-CN].flat_map do |language|
      %i[actor event].flat_map { |axis| rows(axis, language, "#{language}:#{axis}:0") }
    end
  end

  def test_hand_gold_every_template_preserves_targets_facts_order_and_unique_ids
    %w[en-US zh-CN].each do |language|
      %i[actor event].each do |axis|
        original = rows(axis, language)
        (0..12).each do |form|
          changed = rows(axis, language, "fixture:0", form)

          assert_equal 32, changed.map { |row| row.fetch("id") }.uniq.size
          assert V09.audit_rows!(changed)
          assert_equal original.map { |row| row.fetch("target") }, changed.map { |row| V09::Corpus.gold!(row) }
          assert_equal original.map { |row| row.dig("world", "facts") }, changed.map { |row| row.dig("world", "facts") }
          assert_equal original.map { |row| V09::Catalog.family(row) }, changed.map { |row| V09::Catalog.family(row) }
        end
      end
    end
    row = V09::Corpus.world_rows(%w[甲同学 乙同学], %w[button letter], language: :"zh-CN", group: "zh", axis: "event", form: 11).first

    assert_equal "已记录的观察结果：甲同学没有按下按钮。已记录的观察结果：甲同学没有寄出信。", row.fetch("state")
    assert_equal "用记录的事实核对这个说法：甲同学按下了按钮。", row.fetch("question")
    assert_equal "no", V09::Corpus.gold!(row)
    assert_raises(RuntimeError) { V09::Corpus.gold!(row.merge("target" => "yes")) }
    V09::Catalog::POOLS.values.combination(2) { |a, b| assert_empty a & b }
    V09::Catalog::FORMS.values.combination(2) { |a, b| assert_empty a & b }
    legacy = EasyAI::Decision::Data::FactualContrasts::EVENTS.values + DecisionEventGeneralization::Catalog::EVENTS.values

    assert_empty V09::Catalog::NEW_EVENTS.values.map { |v| v[2] } & legacy.map { |v| v[2] }
  end

  def test_both_templates_are_crossed_with_each_axis_within_every_family
    dev = V09::Corpus.panels(:development, worlds: 4, excluded_families: Set.new, historical_materials: Set.new)
    families = dev.values.flatten.map { |row| V09::Catalog.family(row) }.to_set
    final = V09::Corpus.panels("lockbox", worlds: 4, excluded_families: families, historical_materials: Set.new)

    assert_equal 1792, dev.values.sum(&:size)
    assert_empty families & final.values.flatten.map { |row| V09::Catalog.family(row) }.to_set
    %w[expressions composed].each do |view|
      grouped = dev.fetch(view).group_by { |row| [row.fetch("language"), row.dig("world", "family_id")] }
      grouped.each_value do |items|
        assert_equal [9, 10], items.map { |row| row.dig("world", "form") }.uniq.sort
        assert_equal 64, items.size
        a, b = items.group_by { |row| row.dig("world", "form") }.values

        assert_equal a.map { |row| row.fetch("target") }, b.map { |row| row.fetch("target") }
      end
      V09::Catalog::LANGUAGES.each do |language|
        %w[Actor Event].each do |axis|
          assert_equal [9, 10], dev.fetch(view).select { |row| row.fetch("language") == language && row.fetch("source").end_with?(axis) }.map { |row| row.dig("world", "form") }.uniq.sort
        end
      end
    end
    assert_raises(ArgumentError) { V09::Corpus.panels(:development, worlds: 3, excluded_families: Set.new, historical_materials: Set.new) }
  end

  def test_independent_event_expression_factors_and_exact_canonical_replay_fraction
    raw = training_rows
    examples = raw.group_by { |row| [row.fetch("language"), row.fetch("source"), row.fetch("state")] }.values.flat_map { |unit| unit + unit }.map { |row| EasyAI::Decision::Data::Example.new(row) }
    # Keep two complete units consecutively, as the real balanced sampler does.
    augmentation = V09::Augmentation.new(rows: raw)
    variants = V09::VARIANTS.to_h { |variant| [variant, augmentation.call(examples, seed: 41, variant: variant.to_sym)] }
    variants.each do |variant, changed|
      assert_equal examples.map(&:target), changed.map { |row| V09::Corpus.gold!(row) }
      changes = changed.group_by { |row| [row.fetch("language"), row.fetch("source")] }
      changes.each_value do |bucket|
        canonical = bucket.count { |row| row.dig("world", "form").zero? }

        assert_equal variant == "no_replay" ? 0 : (%w[expressions both].include?(variant) ? bucket.size / 2 : bucket.size), canonical
        event_count = bucket.count { |row| V09::Catalog::POOLS.fetch("train").include?(row.dig("world", "query_event")) }

        assert_equal %w[events both no_replay].include?(variant) ? bucket.size / 2 : 0, event_count
      end
    end

    assert_equal variants["events"].map { |row| row.dig("world", "facts") }, variants["both"].map { |row| row.dig("world", "facts") }
    assert_equal variants["both"].map { |row| row.dig("world", "facts") }, variants["no_replay"].map { |row| row.dig("world", "facts") }
    assert_equal variants["expressions"].map { |row| row.dig("world", "form") }, variants["both"].map { |row| row.dig("world", "form") }
    assert_equal variants["both"], augmentation.call(examples, seed: 41, variant: "both")
    assert_raises(ArgumentError) { augmentation.call(examples.first(3), seed: 41, variant: :both) }
    assert_raises(ArgumentError) { augmentation.call(examples, seed: 41, variant: :bad) }
  end

  def test_per_template_metrics_and_independent_pairs_reject_actor_blind_shortcut
    raw = [9, 10].flat_map { |form| %i[actor event].flat_map { |axis| rows(axis, "en-US", "#{axis}:0", form) } }
    logits = raw.map { |row| row.fetch("target") == "yes" ? [1.0, -1.0] : [-1.0, 1.0] }
    metric = V09.metrics(raw, logits)

    assert_equal %w[9 10], metric.fetch("by_form").keys
    assert_in_delta 1, metric.dig("by_language", "en-US", "balanced")
    metric.fetch("by_form").each_value do |panel|
      assert_equal 8, panel.dig("by_language", "en-US", "actor_binding_groups")
      assert_equal 8, panel.dig("by_language", "en-US", "event_binding_groups")
      assert_in_delta 1, panel.dig("by_language", "en-US", "actor_binding_all_correct")
    end
    blind = V09.metrics(raw, Array.new(raw.size) { [1.0, -1.0] })

    assert_in_delta 0.5, blind.dig("by_language", "en-US", "balanced")
    assert_in_delta 0, blind.dig("by_form", "10", "by_language", "en-US", "actor_binding_all_correct")
    assert_raises(RuntimeError) { V09.audit_rows!(raw.drop(1)) }
  end

  def test_reference_matches_old_recipe_and_both_resume_is_exact_with_family_guard
    Dir.mktmpdir do |root|
      raw = training_rows
      dataset = write_dataset(File.join(root, "train.jsonl"), raw)
      config = tiny_config(input: { state_max_tokens: 256, question_option_max_tokens: 256 }, training: { choice_microbatch: 16, gradient_accumulation: 2, steps: 2, track_coverage: true })
      a = EasyAI::Decision::ChoiceModel.new(config)
      b = EasyAI::Decision::ChoiceModel.new(config)
      b.load_state_dict(a.state_dict)
      old = DecisionBindingGeneralization::Trainer.new(model: a, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(root, "old"), variant: :combined)
      fresh = V09::Trainer.new(model: b, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(root, "reference"), variant: "reference")
      old.train
      fresh.train

      assert a.state_dict.all? { |key, tensor| (tensor - b.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal old.state.fetch("coverage"), fresh.state.fetch("coverage")
      trial = V09::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(root, "both"), variant: :both)
      trial.train(steps: 1)
      resumed = V09::Trainer.resume(trial.last_checkpoint, dataset: dataset, output: File.join(root, "resumed"), variant: "both")
      trial.train
      resumed.train

      assert trial.model.state_dict.all? { |key, tensor| (tensor - resumed.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
      assert_equal trial.state.fetch("rendering_coverage"), resumed.state.fetch("rendering_coverage")
      assert_equal trial.state.fetch("coverage"), resumed.state.fetch("coverage")
      assert_raises(ArgumentError) { V09::Trainer.resume(trial.last_checkpoint, dataset: dataset, output: File.join(root, "invalid"), variant: :events) }
      assert_raises(ArgumentError) { V09::Trainer.resume(trial.last_checkpoint, dataset: dataset, output: File.join(root, "changed"), variant: :both, forbidden_families: ["changed"]) }
      examples = raw.first(4).map { |row| EasyAI::Decision::Data::Example.new(row) }
      rendered = fresh.render_batch(examples, seed: 41)
      family = V09::Catalog.family(rendered.first)
      guarded = V09::Trainer.new(model: EasyAI::Decision::ChoiceModel.new(config), tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(root, "guard"), variant: :reference, forbidden_families: [family])

      refute_equal family, V09::Catalog.family(guarded.render_batch(examples, seed: 41).first)
      assert_equal examples.map(&:target), guarded.render_batch(examples, seed: 41).map { |row| V09::Corpus.gold!(row) }
    end
  end

  def test_direction_guards_each_template_axis_language_and_old_pair_retention
    make = lambda do |pair, balanced|
      panel = { "by_language" => V09::Catalog::LANGUAGES.to_h { |language| [language, { "balanced" => balanced, "actor_binding_all_correct" => pair, "event_binding_all_correct" => pair }] }, "worst_language_balanced" => balanced, "nll" => 0.2 }
      metric = panel.merge("by_form" => { "0" => Marshal.load(Marshal.dump(panel)) })
      development = V09::Catalog::VIEWS.to_h { |view| [view, Marshal.load(Marshal.dump(metric))] }
      development["expressions"]["by_form"] = %w[9 10].to_h { |form| [form, Marshal.load(Marshal.dump(panel))] }
      { "development" => development, "regression" => { "expressions" => Marshal.load(Marshal.dump(panel)) } }
    end
    reference = make.call(0.25, 0.85)
    candidate = make.call(0.30, 0.82)

    assert V09.direction?(candidate, reference)
    refute V09.direction?(make.call(0.29, 0.85), reference)
    candidate["development"]["expressions"]["by_form"]["10"]["by_language"]["zh-CN"]["actor_binding_all_correct"] = 0.19

    refute V09.direction?(candidate, reference)
    candidate = make.call(0.30, 0.85)
    candidate["regression"]["expressions"]["by_language"]["en-US"]["event_binding_all_correct"] = 0.19

    refute V09.direction?(candidate, reference)
    Dir.mktmpdir do |root|
      hashes = V09::VARIANTS.to_h do |variant|
        path = File.join(root, "arms", "#{variant}-1337", "result.json")
        V09.seal(path, { "variant" => variant })
        [variant, V09.sha(path)]
      end
      V09.seal(File.join(root, "selection.json"), { "improving" => false, "winner" => nil, "results_sha256" => hashes })

      refute V09.selection(root).fetch("improving")
      assert_raises(RuntimeError) { V09.evaluate_views(root, nil, nil, ["lockbox-events"], "test") }
      File.write(File.join(root, "arms/reference-1337/result.json"), "changed")

      assert_raises(RuntimeError) { V09.selection(root) }
    end
  end
end
