require "test_helper"
require_relative "../../../benchmarks/decision/natural"
require_relative "../../../benchmarks/decision/natural_evaluation"

class NaturalExperimentTest < Minitest::Test
  def test_choice_distribution_separates_predicted_concentration_from_gold_frequency
    first = example
    second = EasyAI::Decision::Data::Example.new(first.to_h.merge("id" => "second", "target" => first.options.last.fetch("id")))
    distribution = DecisionNaturalEvaluation.choice_distribution([first, second], [[-1.0, 1.0], [-1.0, 1.0]])

    assert_equal first.options.last.fetch("id"), distribution.fetch("dominant_id")
    assert_in_delta 1.0, distribution.fetch("dominant_prediction_fraction"), 1e-9
    assert_in_delta 0.5, distribution.fetch("dominant_target_fraction"), 1e-9
  end

  def test_evaluation_requires_every_fixed_budget_to_finish
    Dir.mktmpdir do |directory|
      DecisionNatural::MIXTURES.product(DecisionNatural::POSITIONS).each do |mixture, position|
        path = File.join(directory, "#{mixture}-#{position}")
        FileUtils.mkdir_p(path)
        File.write(File.join(path, "summary.json"), JSON.generate("step" => DecisionNatural::STEPS))
      end
      DecisionNaturalEvaluation.assert_complete(directory)
      File.write(File.join(directory, "broad-rotary/summary.json"), JSON.generate("step" => DecisionNatural::STEPS - 1))

      assert_raises(ArgumentError) { DecisionNaturalEvaluation.assert_complete(directory) }
    end
  end

  def test_macro_weights_sources_equally_then_languages_within_source
    tasks = { "one/en" => { "accuracy" => 1.0 }, "two/en" => { "accuracy" => 0.0 }, "two/zh" => { "accuracy" => 0.0 } }

    assert_in_delta 0.5, DecisionNaturalEvaluation.source_macro(tasks, "accuracy"), 1e-9
  end

  def test_unrepresented_labels_are_explicit_not_assumed_correct_or_incorrect
    first = example
    second = EasyAI::Decision::Data::Example.new(first.to_h.merge("id" => "second", "group_id" => "second"))
    metrics = DecisionNaturalEvaluation.task_metrics([first, second], [[5.0, -5.0], [5.0, -5.0]], EasyAI::Decision::Calibrator.new).values.first

    assert_equal [first.options.last.fetch("id")], metrics.fetch("unrepresented_labels")
    assert_in_delta 1.0, metrics.fetch("balanced_accuracy_observed_labels"), 1e-9
    assert_equal 2, metrics.fetch("independent_groups")
  end

  def test_prediction_artifact_retains_raw_logits_and_calibrated_probabilities
    Dir.mktmpdir do |directory|
      path = File.join(directory, "predictions.jsonl")
      first = example
      logits = [[-2.0, 2.0]]
      DecisionNaturalEvaluation.write_predictions(path, [first], logits, EasyAI::Decision::Calibrator.new(temperature: 2.0))
      saved = JSON.parse(File.read(path))

      assert_equal logits.first, saved.fetch("logits")
      assert_equal first.options.last.fetch("id"), saved.fetch("prediction")
      refute saved.fetch("correct")
      assert_in_delta 1.0, saved.fetch("probabilities").sum, 1e-9
      assert_operator saved.fetch("probabilities").last, :>, 0.5
    end
  end

  def test_position_arms_have_identical_other_configuration
    first = DecisionNatural.config("sinusoidal").to_h
    second = DecisionNatural.config("rotary").to_h
    first.fetch("model").delete("position_encoding")
    second.fetch("model").delete("position_encoding")

    assert_equal first, second
    assert_equal 32, first.dig("training", "choice_microbatch") * first.dig("training", "gradient_accumulation")
    refute first.dig("model", "evidence_head")
  end

  def test_candidate_grouping_preserves_mean_ce_and_gradients_without_dropout
    Dir.mktmpdir do |directory|
      cfg = tiny_config(model: { dropout: 0.0 })
      first = example
      second = EasyAI::Decision::Data::Example.new(first.to_h.merge("id" => "second", "group_id" => "second",
        "options" => first.options + [{ "id" => "other", "text" => "other" }]))
      path = File.join(directory, "data.jsonl")
      File.write(path, [first, second].map { |row| JSON.generate(row.to_h) }.join("\n") + "\n")
      dataset = EasyAI::Decision::Data::Dataset.new(path)
      original = EasyAI::Decision::ChoiceModel.new(cfg)
      grouped = EasyAI::Decision::ChoiceModel.new(cfg)
      grouped.load_state_dict(original.state_dict)
      tokenizer = EasyAI::Tokenizers::ByteBpe.new
      plain = DecisionFittingTrainer.new(model: original, tokenizer: tokenizer, dataset: dataset, device: "cpu", output: File.join(directory, "plain"))
      optimized = EasyAI::Decision::CandidateTrainer.new(model: grouped, tokenizer: tokenizer, dataset: dataset, device: "cpu", output: File.join(directory, "grouped"))
      first_loss = plain.send(:loss_for, [first, second], seed: 123, teacher: true)
      second_loss = optimized.send(:loss_for, [first, second], seed: 123, teacher: true)
      first_loss.backward
      second_loss.backward

      assert_in_delta first_loss.item, second_loss.item, 1e-6
      assert_in_delta first_loss.item, optimized.instance_variable_get(:@batch_losses).fetch("choice_loss"), 1e-6
      assert_equal plain.send(:input_token_count), optimized.send(:input_token_count)
      counts = plain.instance_variable_get(:@collator).input_token_counts

      assert_equal counts[0] * 2 + counts[1] * 3, DecisionNaturalEvaluation.reconstructed_tokens(path, [2, 3], tokenizer)
      original.named_parameters.each { |name, parameter| assert_tensor_close parameter.grad, grouped.named_parameters.fetch(name).grad, 1e-5 }
    end
  end
end
