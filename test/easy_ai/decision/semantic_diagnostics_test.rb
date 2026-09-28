require "test_helper"

class SemanticDiagnosticsTest < Minitest::Test
  def test_diagnosis_detects_dependence_on_both_inputs_within_each_source
    Dir.mktmpdir do |dir|
      reference = write_dataset(File.join(dir, "train.jsonl"), rows("train"))
      validation = write_dataset(File.join(dir, "validation.jsonl"), rows("validation"))
      predictor = Object.new
      predictor.define_singleton_method(:logits) do |row|
        yes = row.state == row.question
        row.options.map { |option| (option["id"] == "yes") == yes ? 4.0 : -4.0 }
      end
      result = EasyAI::Decision::SemanticDiagnostics.new(predictor: predictor, validation: validation, reference: reference).evaluate
      result.fetch("by_source_language").each_value do |stratum|
        conditions = stratum.fetch("conditions")

        assert_in_delta 1.0, conditions["original"]["accuracy"]
        assert_operator conditions["shuffled_state"]["accuracy"], :<, 0.8
        assert_operator conditions["shuffled_question"]["accuracy"], :<, 0.8
        assert_in_delta 0.5, conditions["label_frequency"]["accuracy"]
      end
    end
  end

  def test_diagnosis_rejects_reference_leakage
    Dir.mktmpdir do |dir|
      data = write_dataset(File.join(dir, "data.jsonl"), rows("shared"))

      assert_raises(ArgumentError) do
        EasyAI::Decision::SemanticDiagnostics.new(predictor: nil, validation: data, reference: data)
      end
    end
  end

  private

  def rows(split)
    %w[a b].flat_map do |source|
      Array.new(40) do |i|
        state = i.even? ? "red" : "blue"
        question = i % 4 < 2 ? "red" : "blue"
        id = "#{split}-#{source}-#{i}"
        { id: id, group_id: id, language: "en", source: source, state: state, question: question,
          options: [{ id: "yes", text: "yes" }, { id: "no", text: "no" }], target: state == question ? "yes" : "no" }
      end
    end
  end
end
