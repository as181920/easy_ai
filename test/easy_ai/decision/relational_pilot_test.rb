require_relative "../../test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/relational_encoding"

class RelationalPilotTest < Minitest::Test
  def rules
    { "binary_balanced" => 0.75, "joint_known_recall" => 0.7, "joint_unknown_recall" => 0.7,
      "mixed_groups" => 0.5, "parent_gain" => 0.05, "natural_minimum_accuracy" => 0.65, "natural_tolerance" => 0.03 }
  end

  def metrics(balanced: 0.8, natural: 0.8)
    cell = { "balanced" => balanced, "binding_groups" => 8, "binding_all_correct" => 0.75,
      "actor_binding_groups" => 4, "event_binding_groups" => 4, "actor_binding_all_correct" => 0.75, "event_binding_all_correct" => 0.75,
      "recall" => { "yes" => balanced, "no" => balanced, "unknown" => balanced } }
    { "controlled" => { "by_language" => { "en-US" => cell.dup, "zh-CN" => cell.dup } },
      "natural" => { "by_language" => %w[en-US zh-CN].to_h { |language| [language, { "accuracy" => natural }] } } }
  end

  def test_publication_requires_natural_accuracy_parent_retention_and_gain_in_both_languages
    value = metrics
    baseline = metrics(balanced: 0.65, natural: 0.7)

    assert DecisionRelational.pilot_eligible?(value, baseline, "binary", rules)
    value.fetch("natural").fetch("by_language").fetch("zh-CN")["accuracy"] = 0.6

    refute DecisionRelational.pilot_eligible?(value, baseline, "binary", rules)
    value = metrics
    baseline.fetch("natural").fetch("by_language").fetch("zh-CN")["accuracy"] = 0.9

    refute DecisionRelational.pilot_eligible?(value, baseline, "binary", rules)
    refute DecisionRelational.pilot_eligible?(value, metrics(balanced: 0.79, natural: 0.7), "binary", rules)
  end

  def test_known_advancement_requires_actor_and_event_mastery_separately
    training = metrics(balanced: 1.0).fetch("controlled")
    training.fetch("by_language").each_value { |cell| cell["actor_binding_all_correct"] = 1.0; cell["event_binding_all_correct"] = 1.0 }
    development = metrics.fetch("controlled")
    threshold = { "known_balanced" => 0.95, "mixed_groups" => 0.9, "dev_balanced" => 0.55 }

    assert DecisionRelational.known_mastery?(training, development, threshold)
    training.fetch("by_language").fetch("zh-CN")["actor_binding_all_correct"] = 0.5

    refute DecisionRelational.known_mastery?(training, development, threshold)
  end

  def test_no_pilot_after_unqualified_fitting_or_without_a_confirmable_profile
    Dir.mktmpdir do |root|
      DecisionRelational.write(File.join(root, "fit-selection.json"), { "winner" => nil })

      assert_raises(RuntimeError) { DecisionRelational.confirmed_fit!(root) }
      FileUtils.mkdir_p(File.join(root, "pilot"))
      DecisionRelational.write(File.join(root, "pilot/selection.json"), { "profile" => nil })
      DecisionRelational.stubs(:pilot_protocol).returns({})
      DecisionRelational.expects(:pilot_baseline).never

      assert_raises(RuntimeError) { DecisionRelational.pilot(root, 2027) }
    end
  end

  def test_natural_panels_exclude_disputes_without_automatic_relabelling
    Dir.mktmpdir do |root|
      base = { "id" => "a", "group_id" => "a", "language" => "en-US", "state" => "record", "question" => "claim", "target" => "yes", "source" => "fixture", "agreement" => true, "excluded" => false }
      rows = [base, base.merge("id" => "disputed", "agreement" => false), base.merge("id" => "excluded", "excluded" => true), base.merge("id" => "unknown", "target" => "unknown")]
      path = File.join(root, "audit.json")
      File.write(path, JSON.generate("decisions" => rows))

      assert_equal ["a:binary"], DecisionRelational.natural_rows(path, "binary").map { |row| row.fetch("id") }
      assert_equal %w[a:joint unknown:joint], DecisionRelational.natural_rows(path, "joint").map { |row| row.fetch("id") }
    end
  end
  def focused_dataset(directory)
    rows = EasyAI::Decision::Data::JudgmentCorpus.splits(counts: { "train" => 4 }).fetch("train")
    %w[en-US zh-CN].each do |language|
      %w[natural MASSIVE-Scenario].each do |source|
        rows << { id: "#{language}-#{source}", group_id: "#{language}-#{source}", language: language,
          state: "record", question: "claim", options: [{ id: "yes", text: "yes" }, { id: "no", text: "no" }], target: "yes", source: source }
      end
    end
    write_dataset(File.join(directory, "data.jsonl"), rows)
  end

  def focused_trainer(directory, dataset, output)
    config = tiny_config(model: { encoding_mode: "joint" }, input: { state_max_tokens: 256, question_option_max_tokens: 256 },
      training: { steps: 2, choice_microbatch: 16, gradient_accumulation: 2 })
    model = EasyAI::Decision::ChoiceModel.new(config)
    DecisionRelational::FocusedTrainer.new(model: model, tokenizer: EasyAI::Tokenizers::ByteBpe.new, dataset: dataset, output: File.join(directory, output))
  end

  def test_focused_sampling_keeps_four_query_units_language_balance_and_staged_replay
    Dir.mktmpdir do |directory|
      dataset = focused_dataset(directory)
      trainer = focused_trainer(directory, dataset, "first")
      selected = trainer.send(:sample_indices, 32, rng: Random.new(1337)).map { |index| dataset[index] }

      assert_equal({ "en-US" => 16, "zh-CN" => 16 }, selected.map(&:language).tally)
      assert selected.each_slice(4).all? { |unit| unit.map(&:state).uniq.size == 1 && unit.map(&:question).uniq.size == 4 && unit.map(&:target).sort == %w[no no yes yes] }
      trainer.state["step"] = 2000
      replay = trainer.send(:sample_indices, 32, rng: Random.new(1337)).map { |index| dataset[index] }

      assert replay.any? { |row| row.source == "natural" }
      assert replay.any? { |row| row.source == "MASSIVE-Scenario" }
    end
  end

  def test_focused_resume_preserves_sampling_and_parameter_updates
    Dir.mktmpdir do |directory|
      dataset = focused_dataset(directory)
      trainer = focused_trainer(directory, dataset, "first")
      trainer.train(steps: 1)
      resumed = DecisionRelational::FocusedTrainer.resume(trainer.last_checkpoint, dataset: dataset, output: File.join(directory, "resumed"))
      trainer.train
      resumed.train

      assert_equal DecisionRelational::FocusedTrainer::SIGNATURE, resumed.state.fetch("focused_sampling")
      assert trainer.model.state_dict.all? { |key, value| (value - resumed.model.state_dict.fetch(key)).abs.max.item < 1e-7 }
    end
  end

  def test_focused_variant_cannot_start_before_a_standard_result
    Dir.mktmpdir do |root|
      assert_raises(RuntimeError) { DecisionRelational.authorize_focused!(root) }
      assert_raises(ArgumentError) { DecisionRelational.pilot_directory(root, "unbounded") }
      assert_equal DecisionRelational.pilot_directory(root, "focused"), DecisionRelational.pilot_directory(root, :focused)
    end
  end
  def test_goal_budget_counts_replayed_updates_instead_of_only_the_largest_step
    Dir.mktmpdir do |root|
      path = File.join(root, "fit-joint-1337/choice/training.jsonl")
      FileUtils.mkdir_p(File.dirname(path))
      File.write(path, [1, 1, 2].map { |step| JSON.generate(step: step) }.join("\n") + "\n")

      assert_equal 3, DecisionRelational.update_ledger(root).values.sum
      assert_equal 3, DecisionRelational.enforce_update_budget!(root, 19_997)
      assert_raises(RuntimeError) { DecisionRelational.enforce_update_budget!(root, 19_998) }
    end
  end
  def test_family_uncertainty_uses_paired_families_and_identical_predictions_show_no_gain
    raw = 4.times.map do |index|
      { "id" => index.to_s, "group_id" => "family-#{index / 2}", "language" => "en-US", "state" => "record", "question" => "claim",
        "options" => [{ "id" => "yes", "text" => "yes" }, { "id" => "no", "text" => "no" }], "target" => index.even? ? "yes" : "no" }
    end
    good = raw.map { |row| row.fetch("target") == "yes" ? [2.0, 0.0] : [0.0, 2.0] }
    bad = good.map(&:reverse)
    improved = DecisionRelational.paired_uncertainty(raw, bad, good, samples: 100).fetch("by_language").fetch("en-US")
    same = DecisionRelational.paired_uncertainty(raw, good, good, samples: 100).fetch("by_language").fetch("en-US")

    assert_equal 2, improved.fetch("families")
    assert_equal [1.0, 1.0], improved.fetch("paired_accuracy_gain_interval95")
    assert_equal [0.0, 0.0], same.fetch("paired_accuracy_gain_interval95")
    assert_raises(ArgumentError) { DecisionRelational.paired_uncertainty(raw, [], good) }
  end

  def test_unqualified_profile_cannot_calibrate_or_open_acceptance
    Dir.mktmpdir do |root|
      FileUtils.mkdir_p(File.join(root, "pilot"))
      DecisionRelational.write(File.join(root, "pilot/selection.json"), { "profile" => nil })
      DecisionRelational.stubs(:pilot_protocol).returns({})

      assert_raises(RuntimeError) { DecisionRelational.confirmed_publication(root, "standard") }
      assert_raises(RuntimeError) { DecisionRelational.calibrate(root) }
      assert_raises(RuntimeError) { DecisionRelational.frozen_acceptance(root, "standard") }
      refute_path_exists File.join(root, "pilot/acceptance-opened.json")
      refute_path_exists File.join(root, "pilot/calibration.json")
    end
  end
  def test_corrective_fitting_selection_preserves_the_original_frozen_choice
    Dir.mktmpdir do |root|
      DecisionRelational.stubs(:verify).returns({})
      original = File.join(root, "fit-selection.json")
      DecisionRelational.write(original, { "winner" => "joint-rotary", "results_sha256" => {} })
      before = DecisionRelational.sha(original)
      %w[joint-rotary-candidate separate-rotary].each do |arm|
        directory = File.join(root, "fit-#{arm}-1337")
        FileUtils.mkdir_p(directory)
        DecisionRelational.write(File.join(directory, "result.json"), { "passed" => false })
      end
      result = DecisionRelational.select_fit(root, selection: :candidate)

      assert_nil result.fetch("winner")
      assert_equal before, DecisionRelational.sha(original)
      assert_raises(RuntimeError) { DecisionRelational.check_confirmation!(root, "joint-rotary-candidate", selection: "candidate") }
      assert_raises(ArgumentError) { DecisionRelational.fit_selection_path(root, "unbounded") }
    end
  end
end
