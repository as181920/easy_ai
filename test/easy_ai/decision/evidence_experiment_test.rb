require "test_helper"
require_relative "../../../benchmarks/decision/evidence_evaluation"

class EvidenceExperimentTest < Minitest::Test
  def test_test_evaluation_is_blocked_until_every_fixed_budget_finishes
    Dir.mktmpdir do |root|
      EvidenceExperiment::ARMS.product(EvidenceExperiment::SEEDS).each do |arm, seed|
        path = File.join(root, "#{arm}-#{seed}")
        FileUtils.mkdir_p(path)
        File.write(File.join(path, "summary.json"), JSON.generate({ "step" => 800 }))
      end
      path = File.join(root, "evidence-3407/summary.json")
      File.write(path, JSON.generate({ "step" => 799 }))

      assert_raises(ArgumentError) { EvidenceEvaluation.assert_complete(root) }
      File.write(path, JSON.generate({ "step" => 800 }))
      EvidenceEvaluation.assert_complete(root)
    end
  end

  def test_paired_recipe_changes_only_the_auxiliary_loss_coefficient
    answer = EvidenceExperiment.config(1337, "answer").to_h
    evidence = EvidenceExperiment.config(1337, "evidence").to_h
    weight = evidence.fetch("training").delete("evidence_loss_weight")
    answer.fetch("training").delete("evidence_loss_weight")

    assert_in_delta(0.2, weight)
    assert_equal answer, evidence
  end

  def test_clearing_parameter_gradients_handles_unused_parameters
    parameter = Torch::NN::Parameter.new(Torch.ones(2))
    optimizer = EasyAI::Optim::AdamW.new({ "optional" => parameter }, learning_rate: 0.1)
    optimizer.zero_grad
    parameter.sum.backward
    optimizer.step
    optimizer.zero_grad
    optimizer.step

    assert_nil parameter.grad
    assert_equal({ "optional" => 1 }, optimizer.steps)
  end

  def test_evidence_indices_preserve_physical_line_numbers
    config = tiny_config(model: { evidence_head: true }, input: { state_max_tokens: 128 })
    row = EasyAI::Decision::Data::Example.new(example.to_h.merge("state" => "\nAlice.\n \nBob.\n", "evidence_index" => 3))
    collator = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: config)
    batch = collator.call([row], with_evidence: true)

    assert_equal [1, 5, batch[:state_ids].shape.last], batch[:sentence_mask].shape
    assert_equal [3], batch[:evidence_targets].to_a
  end
end
