require "test_helper"
require_relative "../../../benchmarks/decision/generalization"

class GeneralizationTest < Minitest::Test
  def test_full_candidate_set_is_independent_of_target
    labels = DecisionGeneralization::SCENARIOS.transform_values(&:last)
    first = DecisionGeneralization.build("test", "en-US", "case", "Some request", "Domain?", labels, "alarm")
    second = DecisionGeneralization.build("test", "en-US", "case", "Some request", "Domain?", labels, "music")

    assert_equal 18, first.fetch("options").size
    assert_equal first.except("target"), second.except("target")
  end

  def test_parallel_languages_share_original_group
    english = DecisionGeneralization.build("test", "en-US", "42", "hello", "Domain?", { "yes" => "yes", "no" => "no" }, "yes")
    chinese = DecisionGeneralization.build("test", "zh-CN", "42", "你好", "领域？", { "yes" => "是", "no" => "否" }, "yes")

    assert_equal english.fetch("group_id"), chinese.fetch("group_id")
    refute_equal english.fetch("id"), chinese.fetch("id")
  end

  def test_overlap_hash_normalizes_whitespace_and_unicode_width_without_semantic_rules
    first = DecisionGeneralization.text_hash("  Ａlice\narrived.  ")

    assert_equal first, DecisionGeneralization.text_hash("Alice arrived.")
    refute_equal first, DecisionGeneralization.text_hash("Alice did not arrive.")
  end
end
