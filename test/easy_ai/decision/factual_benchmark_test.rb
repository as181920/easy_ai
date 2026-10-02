require "test_helper"
require "mocha/minitest"
require_relative "../../../benchmarks/decision/factual"

class FactualBenchmarkTest < Minitest::Test
  def test_new_answer_text_does_not_make_a_seen_question_component_fresh
    rows = [[{ state: "already used" }, "question:1", "test"], [{ state: "new answer" }, "question:1", "test"],
      [{ state: "independent" }, "question:2", "test"]]
    historical = Set[DecisionFactual.material("already used")]
    connected = DecisionFactual.historical_components(rows, historical)
    fresh = rows.reject { |_, group, _| connected.include?(group) }

    assert_equal Set["question:1"], connected
    assert_equal ["independent"], fresh.map { |row, _, _| row.fetch(:state) }
  end

  def test_opened_final_panel_cannot_be_reused
    Dir.mktmpdir do |root|
      File.write(File.join(root, "protocol.json"), JSON.generate("files_sha256" => {}))
      File.write(File.join(root, "confirmation.json"), JSON.generate("confirmation_required" => false))
      %w[ce margin].each do |arm|
        FileUtils.mkdir_p(File.join(root, "#{arm}-1337"))
        File.write(File.join(root, "#{arm}-1337/summary.json"), JSON.generate("step" => 2000, "selected" => nil))
      end
      File.write(File.join(root, "acceptance-opened.json"), "{}")

      error = assert_raises(RuntimeError) { DecisionFactual.evaluate(root) }

      assert_match(/Final panel already opened/, error.message)
    end
  end

  def test_incomplete_pilot_cannot_open_final_candidates
    Dir.mktmpdir do |root|
      File.write(File.join(root, "confirmation.json"), JSON.generate("confirmation_required" => false))
      FileUtils.mkdir_p(File.join(root, "ce-1337"))
      File.write(File.join(root, "ce-1337/summary.json"), JSON.generate("step" => 100, "selected" => nil))

      error = assert_raises(RuntimeError) { DecisionFactual.final_candidates(root) }

      assert_match(/Pilot incomplete/, error.message)
    end
  end

  def test_macro_interval_does_not_weight_a_long_component_as_a_better_source
    rows = 100.times.flat_map do |index|
      Array.new(20) { { "group_id" => "a:#{index}", "source" => "A", "prediction" => "yes", "target" => "yes" } } +
        [{ "group_id" => "b:#{index}", "source" => "B", "prediction" => "no", "target" => "yes" }]
    end
    interval = DecisionFactual.macro_interval(rows, seed: 1337)

    assert_equal [0.5, 0.5], interval
    assert_equal interval, DecisionFactual.macro_interval(rows, seed: 1337)
  end

  def test_clean_acceptance_panel_rejects_content_changes
    Dir.mktmpdir do |root|
      FileUtils.mkdir_p(File.join(root, "data"))
      protocol = File.join(root, "protocol.json")
      panel = File.join(root, "data/acceptance.jsonl")
      File.write(protocol, "{}")
      File.write(panel, "original")
      File.write(File.join(root, "provenance.json"), JSON.generate(
        "original_protocol_sha256" => Digest::SHA256.file(protocol).hexdigest,
        "files_sha256" => { "acceptance.jsonl" => Digest::SHA256.file(panel).hexdigest }))
      DecisionFactual.verify_panels(root)
      File.write(panel, "changed")

      error = assert_raises(RuntimeError) { DecisionFactual.verify_panels(root) }

      assert_match(/Clean panel changed/, error.message)
    end
  end

  def test_raw_and_calibrated_metrics_share_one_model_pass
    raw = EasyAI::Decision::Data::FactualContrasts.new("work", 1, "en-US").rows.first(2)
    evaluator = mock("evaluator")
    evaluator.expects(:collect).once.returns([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    SemanticCoverageEvaluation.expects(:new).once.returns(evaluator)
    metrics = DecisionFactual.measure(nil, nil, raw, "cpu", calibrator: EasyAI::Decision::Calibrator.new(temperature: 2.0), include_raw: true)

    assert_in_delta 1.0, metrics.fetch("accuracy")
    assert_equal metrics.fetch("accuracy"), metrics.fetch("raw").fetch("accuracy")
    assert_equal metrics.fetch("pair_all_correct"), metrics.fetch("raw").fetch("pair_all_correct")
    assert_operator metrics.fetch("nll"), :>, metrics.fetch("raw").fetch("nll")
  end

  def test_controlled_report_separates_uncertainty_from_complete_pairs
    rows = EasyAI::Decision::Data::FactualContrasts.new("work", 1, "en-US").rows.map do |row|
      logits = row.fetch("options").map { |option| option.fetch("id") == row.fetch("target") ? 3.0 : 0.0 }
      row.merge("prediction" => row.fetch("target"), "logits" => logits)
    end
    cells = DecisionFactual.controlled_slices(rows, 1.0).fetch("phenomenon")

    assert_in_delta 1.0, cells.fetch("negative_question/en-US").fetch("pair_all_correct")
    assert_nil cells.fetch("uncertainty/en-US").fetch("pair_all_correct")
    assert_in_delta 1.0, cells.fetch("uncertainty/en-US").fetch("accuracy")
  end
end
