require "test_helper"
require_relative "../../../benchmarks/decision/fitting"

class FittingTest < Minitest::Test
  def test_positional_comparison_changes_only_position_function
    sinusoidal = DecisionFitting.config("sinusoidal").to_h
    rotary = DecisionFitting.config("rotary").to_h
    sinusoidal.fetch("model").delete("position_encoding")
    rotary.fetch("model").delete("position_encoding")

    assert_equal sinusoidal, rotary
    assert_in_delta 0.0, sinusoidal.dig("model", "dropout")
    refute sinusoidal.dig("model", "evidence_head")
    assert_equal 2000, sinusoidal.dig("training", "steps")
  end

  def test_candidate_permutations_preserve_target_and_every_candidate
    rows = [example, example]
    shuffled = DecisionFittingTrainer.permute(rows, 15)

    assert_equal shuffled.map(&:to_h), DecisionFittingTrainer.permute(rows, 15).map(&:to_h)
    assert_equal rows.map(&:target), shuffled.map(&:target)
    assert_equal rows.map { |row| row.options.sort_by { |option| option.fetch("id") } }, shuffled.map { |row| row.options.sort_by { |option| option.fetch("id") } }
    assert_equal rows.map(&:state), shuffled.map(&:state)
  end

  def test_positional_switch_preserves_tensors_but_changes_forward
    config = tiny_config(model: { position_scale: 0.02, dropout: 0.0 })
    Torch.manual_seed(1337)
    sinusoidal = EasyAI::Decision::ChoiceModel.new(config).eval
    rotary = EasyAI::Decision::ChoiceModel.new(config.with(model: { position_encoding: "rotary" })).eval
    rotary.load_state_dict(sinusoidal.state_dict)
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: config).call([example])

    assert_equal sinusoidal.parameter_count, rotary.parameter_count
    sinusoidal.state_dict.each { |name, value| assert_tensor_close value, rotary.state_dict.fetch(name) }
    refute_equal sinusoidal.call(batch).to_a, rotary.call(batch).to_a
  end

  def test_same_device_evaluation_preserves_optimizer_parameters_and_training
    config = tiny_config(model: { dropout: 0.0 })
    model = EasyAI::Decision::ChoiceModel.new(config)
    tokenizer = EasyAI::Tokenizers::ByteBpe.new
    optimizer = EasyAI::Optim::AdamW.new(model.named_parameters, learning_rate: 0.001)
    references = model.named_parameters.transform_values(&:object_id)
    before = model.state_dict.to_h { |name, tensor| [name, tensor.clone] }
    evaluator = SemanticCoverageEvaluation.new(model: model, tokenizer: tokenizer, device: "cpu")
    evaluator.collect([example])
    model.train
    batch = EasyAI::Decision::Data::Collator.new(tokenizer: tokenizer, config: config).call([example])
    model.call(batch).sum.backward
    optimizer.step

    assert_equal references, model.named_parameters.transform_values(&:object_id)
    assert_operator optimizer.steps.size, :>, 0
    assert model.state_dict.any? { |name, tensor| (tensor - before.fetch(name)).abs.max.item > 0 }
  end

  def test_counterfactual_groups_require_every_member_correct
    raw = 4.times.map { |i| { "world" => { "checks" => { "fact_flip" => (i / 2).to_s, "binding" => "one" } } } }
    result = DecisionFitting.groups(raw, [true, true, true, false])

    assert_in_delta 0.5, result.dig("fact_flip", "all_correct")
    assert_in_delta 0.0, result.dig("binding", "all_correct")
    assert_equal 2, result.dig("fact_flip", "count")
  end
end
