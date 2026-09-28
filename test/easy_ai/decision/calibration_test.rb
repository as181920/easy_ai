require "test_helper"

class CalibrationTest < Minitest::Test
  def test_temperature_reduces_overconfident_nll_without_changing_rank
    logits = [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]]
    targets = [0, 1, 1, 1]
    original = EasyAI::Decision::Calibrator.new
    fitted = EasyAI::Decision::Calibrator.new.fit(logits, targets)

    assert_operator fitted.nll(logits, targets), :<, original.nll(logits, targets)
    assert_operator fitted.temperature, :>, 1
    assert_operator fitted.probabilities(logits.first)[0], :>, fitted.probabilities(logits.first)[1]
  end

  def test_extreme_finite_logits_have_normalized_probabilities
    probabilities = EasyAI::Decision::Calibrator.new.probabilities([10_000.0, 9999.0, -10_000.0])

    assert probabilities.all?(&:finite?)
    assert_in_delta 1.0, probabilities.sum, 1e-12
  end

  def test_invalid_temperatures_and_logits_are_rejected
    assert_raises(ArgumentError) { EasyAI::Decision::Calibrator.new(temperature: 0) }
    assert_raises(ArgumentError) { EasyAI::Decision::Calibrator.new.probabilities([Float::NAN, 0]) }
  end

  def test_perfect_predictions_have_correct_metrics
    result = EasyAI::Decision::Evaluator.metrics([[20.0, -20.0], [-20.0, 20.0]], [0, 1])

    assert_in_delta(1.0, result["accuracy"])
    assert_operator result["brier"], :<, 1e-12
    assert_operator result["ece"], :<, 1e-12
  end
end
