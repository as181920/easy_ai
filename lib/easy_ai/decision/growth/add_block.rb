module EasyAI
  module Decision
    module Growth
      class AddBlock
        def self.apply(model)
          old = model.config
          widths = old[:model]["ffn_sizes"] || Array.new(old[:model]["encoder_layers"], old[:model]["ffn_size"])
          config = old.with(model: { encoder_layers: widths.length + 1, ffn_sizes: widths + [old[:model]["ffn_size"]] })
          grown = ChoiceModel.new(config)
          state = grown.state_dict
          Torch.no_grad { model.state_dict.each { |name, tensor| state.fetch(name).copy!(tensor.detach.cpu) } }
          grown.load_state_dict(state)
          grown.encoder.blocks[widths.length].identity!
          grown
        end
      end
    end
  end
end
