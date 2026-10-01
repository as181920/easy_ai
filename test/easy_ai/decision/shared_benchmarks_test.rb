require "test_helper"
require "open3"
require_relative "../../../benchmarks/decision/shared_benchmarks"

class SharedBenchmarksTest < Minitest::Test
  def jev(spec, labels, target)
    { "id" => "case", "group" => "pair", "split" => "public", "provenance" => {}, "family" => "policy",
      "state" => { "z" => [true, "你好"], "a" => 1 }, "question" => spec, "labels" => labels, "expected" => target }
  end

  def test_boolean_criteria_and_canonical_state_preserve_wire_labels
    row = EasyAI::Decision::Data::BenchmarkAdapter.jev(jev({ "type" => "noul", "instructions" => "Allowed?",
      "criteria" => { "false" => "prohibited", "true" => "permitted" } }, %w[no yes], "yes"), tier: "original")

    assert_equal [{ "id" => "no", "text" => "false: prohibited" }, { "id" => "yes", "text" => "true: permitted" }], row.fetch("options")
    assert_equal '{"a": 1, "z": [true, "你好"]}', row.fetch("state")
    assert_equal "yes", row.fetch("target")
  end

  def test_score_keeps_every_level_when_target_changes
    source = jev({ "type" => "score", "instructions" => "Rate", "criteria" => %w[low medium high] }, %w[0 1 2], "0")
    adapter = EasyAI::Decision::Data::BenchmarkAdapter
    first = adapter.jev(source, tier: "easy")
    second = adapter.jev(source.merge("expected" => "2"), tier: "easy")

    assert_equal %w[0 1 2], first.fetch("options").map { |option| option.fetch("id") }
    assert_equal "2: high", first.fetch("options").last.fetch("text")
    assert_equal first.except("target"), second.except("target")
  end

  def typed
    questions = 5.times.to_h { |i| ["q#{i}", { "type" => "choice", "instructions" => "Choose", "criteria" => { "a" => "First", "b" => "Second" } }] }
    gold = questions.to_h { |name, _| [name, { "label" => "a", "probabilities" => { "a" => 0.7, "b" => 0.3 } }] }
    { "id" => "typed-case", "split" => "test", "workflow" => "workflow", "state" => "Visible state", "questions" => JSON.generate(questions),
      "gold" => JSON.generate(gold), "factors" => "SECRET", "label_agreement" => "SECRET" }
  end

  def test_typed_gold_and_factors_never_enter_model_inputs
    rows = EasyAI::Decision::Data::BenchmarkAdapter.typed(typed)
    changed = typed.merge("factors" => "OTHER", "gold" => typed.fetch("gold").gsub('"label":"a"', '"label":"b"'))
    others = EasyAI::Decision::Data::BenchmarkAdapter.typed(changed)
    inputs = ->(items) { items.map { |row| EasyAI::Decision::Data::Example.new(row).to_h.slice("state", "question", "options") } }

    assert_equal 5, rows.size
    assert_equal 1, rows.map { |row| row.fetch("group_id") }.uniq.size
    assert_equal inputs.call(rows), inputs.call(others)
    refute_includes JSON.generate(inputs.call(rows)), "SECRET"
  end

  def test_typed_rejects_training_split_and_mismatched_reference_labels
    adapter = EasyAI::Decision::Data::BenchmarkAdapter

    assert_raises(ArgumentError) { adapter.typed(typed.merge("split" => "train")) }
    assert_raises(ArgumentError) { adapter.typed(typed.merge("gold" => typed.fetch("gold").gsub('"b":0.3', '"c":0.3'))) }
  end

  def decision(id, probs, target: "a", reference: nil)
    { "id" => id, "group_id" => "case", "target" => target, "labels" => %w[a b], "probabilities" => probs,
      "correct" => true, "error" => "input_limit", "benchmark" => reference ? { "reference_probabilities" => reference } : {} }
  end

  def test_failures_stay_in_denominator_and_invalidate_case_group
    rows = [decision("first", { "a" => 0.8, "b" => 0.2 }), decision("second", nil)]
    result = DecisionSharedBenchmarks.metrics(rows)

    assert_in_delta(0.5, result.fetch("accuracy"))
    assert_in_delta(0.5, result.fetch("coverage"))
    assert_in_delta(1.0, result.fetch("supported_accuracy"))
    assert_equal({ "input_limit" => 1 }, result.fetch("errors"))
    assert_in_delta(0.0, result.fetch("groups").fetch("all_correct"))
  end

  def test_invalid_vectors_fail_closed_even_if_correct_flag_is_true
    vectors = [{ "a" => Float::NAN, "b" => 0.5 }, { "a" => 0.6, "b" => 0.6 }, { "a" => 1.0 }, { "a" => "1", "b" => 0 }]
    result = DecisionSharedBenchmarks.metrics(vectors.each_with_index.map { |probs, i| decision(i.to_s, probs) })

    assert_equal 0, result.fetch("supported")
    assert_in_delta(0.0, result.fetch("accuracy"))
    assert_equal({ "invalid_vector" => 4 }, result.fetch("errors"))
    refute result.key?("probability_quality_supported_only")
  end

  def test_prediction_uses_lexical_tie_break_and_recomputes_correctness
    result = DecisionSharedBenchmarks.metrics([decision("tie", { "b" => 0.5, "a" => 0.5 }, target: "b")])

    assert_equal "a", DecisionSharedBenchmarks.prediction({ "b" => 0.5, "a" => 0.5 })
    assert_in_delta(0.0, result.fetch("accuracy"))
  end

  def test_probability_quality_matches_one_hot_and_soft_reference_formulas
    probs = { "a" => 0.75, "b" => 0.25 }
    hard = DecisionSharedBenchmarks.probability_quality([decision("hard", probs)])
    soft = DecisionSharedBenchmarks.probability_quality([decision("soft", probs, reference: probs)])

    assert_in_delta(-Math.log(0.75), hard.fetch("reference_cross_entropy"), 1e-12)
    assert_in_delta 0.125, hard.fetch("reference_brier_sum"), 1e-12
    assert_in_delta 0.0, soft.fetch("reference_kl"), 1e-12
    assert_in_delta 0.0, soft.fetch("reference_brier_sum"), 1e-12
    assert_in_delta(-0.75 * Math.log(0.75) - 0.25 * Math.log(0.25), soft.fetch("reference_cross_entropy"), 1e-12)
  end

  def test_invalid_reference_is_rejected_and_empty_slice_is_explicit
    assert_raises(ArgumentError) { DecisionSharedBenchmarks.metrics([]) }
    assert_raises(ArgumentError) do
      DecisionSharedBenchmarks.probability_quality([decision("bad", { "a" => 0.5, "b" => 0.5 }, reference: { "a" => 1.0 })])
    end
  end

  def test_download_checks_cache_without_network_and_rejects_changed_snapshots
    Dir.mktmpdir do |directory|
      name = "jev-original.jsonl"
      path = File.join(directory, name)
      File.write(path, "cached")
      capture_io { DecisionSharedBenchmarks.fetch_snapshot(name, Digest::SHA256.hexdigest("cached"), directory) }

      assert_equal "cached", File.read(path)
      assert_raises(ArgumentError) { DecisionSharedBenchmarks.fetch_snapshot(name, "bad", directory) }
    end
  end

  def test_download_urls_pin_jev_revision_and_test_page_offset
    assert_includes DecisionSharedBenchmarks.source_url("jev-original.jsonl"), DecisionSharedBenchmarks::REVISION
    assert_includes DecisionSharedBenchmarks.source_url("typed-003.json"), "offset=300&length=100"
  end

  def test_absolute_cli_evaluation_does_not_reenter_preparation
    Dir.mktmpdir do |directory|
      path = File.expand_path("../../../benchmarks/decision/shared_benchmarks.rb", __dir__)
      _output, error, status = Open3.capture3(RbConfig.ruby, path, "--phase", "evaluate", "--output", directory)

      refute_predicate status, :success?
      assert_includes error, "answer-1337/summary.json"
      refute_path_exists File.join(directory, "shared-benchmarks")
    end
  end
end
