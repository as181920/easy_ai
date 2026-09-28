require "test_helper"

class StateAblationTest < Minitest::Test
  def test_diagnostic_exposes_a_predictor_that_ignores_state
    Dir.mktmpdir do |dir|
      reference = write_dataset(File.join(dir, "train.jsonl"), [example(id: "train")])
      validation = write_dataset(File.join(dir, "validation.jsonl"), [example(id: "v1"), example(id: "v2", state: "blue", target: "b")])
      predictor = Object.new
      predictor.define_singleton_method(:logits) { |_example| [1.0, 0.0] }
      report = EasyAI::Decision::StateAblation.new(predictor: predictor, validation: validation, reference: reference).evaluate
      conditions = report.fetch("conditions")

      assert_in_delta 1.0, conditions["shuffled_state"]["prediction_agreement_with_original"]
      assert_in_delta 0.0, conditions["constant_state"]["mean_max_probability_change"]
      assert_in_delta 0.5, conditions["original"]["accuracy"]
      assert_in_delta 0.5, report["expected_random_accuracy"]
    end
  end

  def test_diagnostic_rejects_overlapping_source_groups
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "train.jsonl"), [example])

      assert_raises(ArgumentError) { EasyAI::Decision::StateAblation.new(predictor: Object.new, validation: data, reference: data) }
    end
  end
end
