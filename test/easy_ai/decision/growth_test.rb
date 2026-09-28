require "test_helper"

class GrowthTest < Minitest::Test
  def setup
    Torch.manual_seed(6)
    @model = EasyAI::Decision::ChoiceModel.new(tiny_config(model: { embedding_norm: true, position_scale: 0.02, score_mode: "matching" })).eval
    @batch = EasyAI::Decision::Data::Collator.new(tokenizer: EasyAI::Tokenizers::ByteBpe.new, config: tiny_config).call([example])
  end

  def test_new_residual_block_preserves_outputs
    grown = EasyAI::Decision::Growth::AddBlock.apply(@model).eval

    assert_equal 2, grown.config[:model]["encoder_layers"]
    assert_tensor_close @model.call(@batch), grown.call(@batch)
  end

  def test_wider_ffn_preserves_outputs_and_can_learn
    grown = EasyAI::Decision::Growth::WidenFfn.apply(@model, layer: 0, size: 48).eval

    assert_tensor_close @model.call(@batch), grown.call(@batch)
    Torch::NN::Functional.cross_entropy(grown.call(@batch), @batch[:targets]).backward
    new_columns = grown.encoder.blocks[0].ffn.down.weight.grad.narrow(1, 32, 16)

    assert_operator new_columns.abs.sum.item, :>, 0
  end

  def test_optimizer_moments_are_preserved_and_extended
    optimizer = EasyAI::Optim::AdamW.new(@model.named_parameters, learning_rate: 0.001)
    Torch::NN::Functional.cross_entropy(@model.call(@batch), @batch[:targets]).backward
    optimizer.step
    saved = optimizer.state_dict
    grown = EasyAI::Decision::Growth::WidenFfn.apply(@model, layer: 0, size: 48)
    restored = EasyAI::Optim::AdamW.new(grown.named_parameters, learning_rate: 0.001).load_state_dict(saved, allow_growth: true)
    key = "encoder.blocks.0.ffn.down.weight/m"
    state = restored.state_dict

    assert_tensor_close saved["tensors"][key], state["tensors"][key].narrow(1, 0, 32)
    assert_in_delta(0.0, state["tensors"][key].narrow(1, 32, 16).abs.sum.item)
    assert_equal optimizer.steps, restored.steps
  end

  def test_plateau_controller_rolls_back_unsuccessful_trial
    config = tiny_config(growth: { enabled: true, patience: 2, trial_evaluations: 1 })
    controller = EasyAI::Decision::Growth::Controller.new(config)

    assert_equal :wait, controller.observe(train_loss: 1.0, validation_loss: 1.0, step: 1, model_config: config)
    assert_equal :grow, controller.observe(train_loss: 1.0, validation_loss: 1.0, step: 2, model_config: config)
    controller.start_trial(path: "/checkpoint", baseline_loss: 1.0, step: 2)

    assert_equal :rollback, controller.observe(train_loss: 0.9, validation_loss: 1.1, step: 3, model_config: config)
    assert_equal "/checkpoint", controller.reject(step: 3)
    assert_nil controller.state["pending"]
  end
end
