require "test_helper"
require_relative "../../../benchmarks/decision/expression_retention"

# Hand-valued counterexamples isolate each guard while every other guard passes.
# rubocop:disable Minitest/MultipleAssertions
class ExpressionRetentionPolicyTest < Minitest::Test
  V09 = DecisionExpressionRetention

  def fixture(pair:, balanced:, nll:)
    panel = { "by_language" => V09::Catalog::LANGUAGES.to_h { |language| [language, { "balanced" => balanced, "actor_binding_all_correct" => pair, "event_binding_all_correct" => pair }] },
      "nll" => nll, "worst_language_balanced" => balanced }
    development = V09::Catalog::VIEWS.to_h do |view|
      forms = view == "expressions" ? %w[9 10] : ["0"]
      [view, Marshal.load(Marshal.dump(panel)).merge("by_form" => forms.to_h { |form| [form, Marshal.load(Marshal.dump(panel))] })]
    end
    { "development" => development, "regression" => { "historical" => Marshal.load(Marshal.dump(panel)) } }
  end

  def comparison
    reference = fixture(pair: 0.70, balanced: 0.85, nll: 0.5)
    candidate = fixture(pair: 0.75, balanced: 0.86, nll: 0.4)
    [[reference, [0.60, 0.20, 0.55, 0.40]], [candidate, [0.65, 0.30, 0.60, 0.45]]].each do |value, pairs|
      keys = V09::Catalog::LANGUAGES.product(%w[actor event])
      keys.zip(pairs).each do |(language, axis), score|
        value["development"]["events"]["by_form"]["0"]["by_language"][language]["#{axis}_binding_all_correct"] = score
      end
    end
    [candidate, reference]
  end

  def test_hand_derived_minimum_mean_and_ranking
    candidate, reference = comparison

    assert_equal 12, V09.target_cells(candidate).size
    assert V09.direction?(candidate, reference)
    assert_in_delta 0.30, V09.rank(candidate)[0], 1e-12
    assert_in_delta 2.0 / 3, V09.rank(candidate)[1], 1e-12
    assert_in_delta 0.86, V09.rank(candidate)[2], 1e-12
    assert_in_delta(-1.6, V09.rank(candidate)[3], 1e-12)
  end

  def test_individual_target_guard_can_fail_while_minimum_improves_ten_points
    %w[actor event].each do |axis|
      candidate, reference = comparison
      cell = candidate["development"]["expressions"]["by_form"]["10"]["by_language"]["en-US"]
      cell["#{axis}_binding_all_correct"] = 0.64

      assert_in_delta 0.30, V09.target_cells(candidate).values.min, 1e-12
      refute V09.direction?(candidate, reference)
      cell["#{axis}_binding_all_correct"] = 0.65

      assert V09.direction?(candidate, reference)
    end
  end

  def test_each_basic_template_balanced_guard_is_independent_of_pair_scores
    [["expressions", "9", "zh-CN"], ["actors", "0", "en-US"]].each do |view, form, language|
      candidate, reference = comparison
      cell = candidate["development"][view]["by_form"][form]["by_language"][language]
      cell["balanced"] = 0.819

      refute V09.direction?(candidate, reference)
      cell["balanced"] = 0.82

      assert V09.direction?(candidate, reference)
    end
  end

  def test_old_regression_row_and_both_pair_guards_are_independent
    { "balanced" => [0.819, 0.82], "actor_binding_all_correct" => [0.64, 0.65], "event_binding_all_correct" => [0.64, 0.65] }.each do |key, (outside, boundary)|
      candidate, reference = comparison
      cell = candidate["regression"]["historical"]["by_language"]["zh-CN"]
      cell[key] = outside

      refute V09.direction?(candidate, reference)
      cell[key] = boundary

      assert V09.direction?(candidate, reference)
    end
  end

  def test_high_mean_does_not_compensate_for_insufficient_minimum_gain
    candidate, reference = comparison
    candidate["development"]["expressions"]["by_form"].each_value do |panel|
      panel["by_language"].each_value { |cell| cell["actor_binding_all_correct"] = cell["event_binding_all_correct"] = 0.99 }
    end
    candidate["development"]["events"]["by_form"]["0"]["by_language"]["en-US"]["event_binding_all_correct"] = 0.24

    assert_operator V09.rank(candidate)[1], :>, 0.8
    refute V09.direction?(candidate, reference)
  end

  def test_composed_stress_remains_a_diagnostic
    candidate, reference = comparison
    candidate["development"]["composed"]["by_form"]["0"]["by_language"].each_value do |cell|
      cell.transform_values! { 0.0 }
    end

    assert V09.direction?(candidate, reference)
    assert_in_delta 0.30, V09.rank(candidate)[0], 1e-12
  end
end
