require "test_helper"

class ReleasePolicyTest < Minitest::Test
  def test_threshold_is_fitted_on_calibration_and_keeps_minimum_coverage
    probabilities = Array.new(300) { |index| index.even? ? [0.95, 0.05] : [0.05, 0.95] }
    targets = Array.new(300) { |index| index % 2 }
    policy = EasyAI::Decision::ReleasePolicy.fit(probabilities, targets)

    assert policy.fetch("available")
    assert_in_delta 1.0, policy.fetch("calibration").fetch("coverage")
    assert_empty EasyAI::Decision::ReleasePolicy.failures(policy.fetch("calibration"), expected_labels: [0, 1])
  end

  def test_an_unreliable_model_has_no_automation_policy
    policy = EasyAI::Decision::ReleasePolicy.fit(Array.new(300) { [0.99, 0.01] }, Array.new(300) { |index| index % 2 })

    refute policy.fetch("available")
    assert_nil policy.fetch("threshold")
  end

  def test_missing_classes_fail_even_when_observed_accuracy_is_perfect
    metrics = EasyAI::Decision::ReleaseMetrics.measure(Array.new(300) { [0.99, 0.01] }, Array.new(300, 0), threshold: 0.9)

    assert_includes EasyAI::Decision::ReleasePolicy.failures(metrics, expected_labels: [0, 1]), "Missing target labels: 1"
  end

  def test_conservative_calibration_guard_trades_coverage_for_accuracy
    probabilities = Array.new(200) { [0.99, 0.01] } + Array.new(400) { [0.75, 0.25] }
    targets = Array.new(555, 0) + Array.new(45, 1)
    ordinary = EasyAI::Decision::ReleasePolicy.fit(probabilities, targets)
    conservative = EasyAI::Decision::ReleasePolicy.fit(probabilities, targets, minimum_accuracy: 0.94, minimum_lower_bound: 0.90)

    assert_in_delta 1.0, ordinary.fetch("calibration").fetch("coverage")
    assert_in_delta 1.0 / 3, conservative.fetch("calibration").fetch("coverage")
    assert_operator conservative.fetch("threshold"), :>, ordinary.fetch("threshold")
  end

  def test_calibration_guards_cannot_weaken_the_fixed_release_requirements
    assert_raises(ArgumentError) { EasyAI::Decision::ReleasePolicy.fit([[0.5, 0.5]], [0], minimum_accuracy: 0.8) }
  end
end
