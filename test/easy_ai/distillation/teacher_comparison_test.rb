require "test_helper"
require_relative "../../../benchmarks/distillation/teacher_comparison"

class TeacherComparisonTest < Minitest::Test
  def test_agreement_does_not_count_as_correctness
    rows = [{ "original" => "no", "reversed" => "no", "target" => "yes" },
      { "original" => "yes", "reversed" => "yes", "target" => "yes" }]
    metrics = TeacherComparison.metrics(rows)

    assert_in_delta 1.0, metrics.fetch("permutation_agreement")
    assert_in_delta 0.5, metrics.fetch("both_correct_rate")
    assert_equal 1, metrics.fetch("original_correct")
  end

  def test_comparison_rebuilds_metrics_from_records_and_reports_regressions
    Dir.mktmpdir do |dir|
      baseline = write_run(File.join(dir, "baseline"), profile: "generic", answers: [0, 1, 0, 1])
      experiment = write_run(File.join(dir, "experiment"), profile: "task_specific", answers: [0, 1, 1, 1])
      report = TeacherComparison.compare(baseline, experiment)

      assert report.dig("baseline", "passed")
      refute report.dig("experiment", "passed")
      assert_in_delta 0.5, report.dig("experiment", "overall", "both_correct_rate")
      assert_equal({ "wrong_to_correct" => 0, "correct_to_wrong" => 1 }, report.dig("transitions", "original"))
      assert_equal "relations", report.fetch("changed_pairs").first.fetch("source")
    end
  end

  def test_comparison_rejects_changed_teacher_or_data
    Dir.mktmpdir do |dir|
      baseline = write_run(File.join(dir, "baseline"), profile: "generic", answers: [0, 1, 0, 1])
      teacher_change = write_run(File.join(dir, "teacher"), profile: "task_specific", answers: [0, 1, 0, 1], revision: "2")
      data_change = write_run(File.join(dir, "data"), profile: "task_specific", answers: [0, 1, 0, 1], state: "different")

      assert_raises(ArgumentError) { TeacherComparison.compare(baseline, teacher_change) }
      assert_raises(ArgumentError) { TeacherComparison.compare(baseline, data_change) }
    end
  end

  private

  def write_run(path, profile:, answers:, revision: "1", state: "red")
    FileUtils.mkdir_p(path)
    rows = %w[BoolQ relations].flat_map do |source|
      row = example(id: "original:#{source}", state: state).to_h.merge("source" => source)
      [row, row.merge("id" => "reversed:#{source}", "options" => row.fetch("options").reverse)]
    end
    data = write_dataset(File.join(path, "development.jsonl"), rows)
    protocol = { "gate" => { "relation_accuracy" => 0.95, "public_source_macro_accuracy" => 0.70, "permutation_agreement" => 0.95 } }
    EasyAI::Distillation::Artifact.write_json(File.join(path, "protocol.json"), protocol)
    teacher = Object.new
    teacher.define_singleton_method(:signature) { { "revision" => revision } }
    teacher.define_singleton_method(:call) { |_request| { "kind" => "text", "text" => JSON.generate("answer" => answers.shift) } }
    EasyAI::Distillation::Collector.new(teacher: teacher, adapter: EasyAI::Decision::DistillationAdapter.new(profile: profile),
      output: File.join(path, "teacher"), purpose: "development", progress: nil).run(data)
    path
  end
end
