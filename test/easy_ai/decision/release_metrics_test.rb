require "test_helper"

class ReleaseMetricsTest < Minitest::Test
  def test_rejecting_everything_cannot_claim_perfect_accuracy
    result = EasyAI::Decision::ReleaseMetrics.measure([[0.6, 0.4], [0.3, 0.7]], [0, 1], threshold: 0.9)

    assert_in_delta(0.0, result.fetch("coverage"))
    assert_nil result.fetch("accepted_accuracy")
    assert_nil result.fetch("accepted_accuracy_lower_95")
  end

  def test_correctness_is_recomputed_and_minority_recall_is_visible
    result = EasyAI::Decision::ReleaseMetrics.measure([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], [0, 0, 1], threshold: 0.8)

    assert_in_delta 2.0 / 3, result.fetch("accuracy")
    assert_in_delta(0.5, result.fetch("balanced_accuracy"))
    assert_in_delta(1.0, result.fetch("accepted_accuracy"))
    assert_operator result.fetch("accepted_accuracy_lower_95"), :<, 0.5
  end

  def test_invalid_vectors_and_targets_are_rejected
    [[[Float::NAN, 0.5], 0], [[0.8, 0.8], 0], [[0.5, 0.5], 2]].each do |vector, target|
      assert_raises(ArgumentError) { EasyAI::Decision::ReleaseMetrics.measure([vector], [target], threshold: 0.8) }
    end
    assert_raises(ArgumentError) { EasyAI::Decision::ReleaseMetrics.measure([], [], threshold: 0.8) }
  end
end
