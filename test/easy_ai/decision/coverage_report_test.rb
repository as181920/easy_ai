require "test_helper"
require_relative "../../../benchmarks/decision/semantic_coverage_report"

class CoverageReportTest < Minitest::Test
  GATE = { "mean_macro_accuracy_gain" => 0.03, "maximum_mean_source_regression" => 0.03, "minimum_improved_seeds" => 2 }.freeze

  def test_one_good_seed_cannot_hide_two_regressions
    before = { 1 => score(0.5, 0.5), 2 => score(0.5, 0.5), 3 => score(0.5, 0.5) }
    after = { 1 => score(0.9, 0.9), 2 => score(0.49, 0.49), 3 => score(0.49, 0.49) }
    result = SemanticCoverageReport.comparison(before, after, GATE)

    assert_operator result.fetch("mean_macro_accuracy_gain"), :>, 0.03
    assert_equal 1, result.fetch("improved_seeds")
    refute result.fetch("passed")
  end

  def test_source_regression_is_not_hidden_by_macro_gain
    before = { 1 => score(0.5, 0.5), 2 => score(0.5, 0.5), 3 => score(0.5, 0.5) }
    after = { 1 => score(0.7, 0.45), 2 => score(0.7, 0.45), 3 => score(0.7, 0.45) }
    result = SemanticCoverageReport.comparison(before, after, GATE)

    assert_operator result.fetch("mean_macro_accuracy_gain"), :>, 0.03
    assert_in_delta(-0.05, result.dig("mean_source_accuracy_changes", "b"))
    refute result.fetch("passed")
    improved = before.transform_values { |_value| score(0.6, 0.6) }

    assert SemanticCoverageReport.comparison(before, improved, GATE).fetch("passed")
  end

  def test_final_challenge_cannot_be_evaluated_before_every_run_finishes
    Dir.mktmpdir do |dir|
      protocol = { "files_sha256" => {}, "arms" => ["baseline"], "seeds" => [1337], "budgets" => [1000, 4000] }
      File.write(File.join(dir, "protocol.json"), JSON.generate(protocol))
      FileUtils.mkdir_p(File.join(dir, "baseline-1337"))
      File.write(File.join(dir, "baseline-1337/summary.json"), JSON.generate("budgets" => [{ "budget" => 1000 }]))

      assert_raises(ArgumentError) { SemanticCoverageReport.protocol(dir) }
    end
  end

  private

  def score(left, right)
    { "macro_source_accuracy" => (left + right) / 2, "nll" => 1.0,
      "by_source" => { "a" => { "accuracy" => left }, "b" => { "accuracy" => right } } }
  end
end
