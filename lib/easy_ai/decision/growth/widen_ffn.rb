module EasyAI
  module Decision
    module Growth
      class WidenFfn
        def self.apply(model, layer:, size:)
          old = model.config
          widths = (old[:model]["ffn_sizes"] || Array.new(old[:model]["encoder_layers"], old[:model]["ffn_size"])).dup
          previous = widths.fetch(layer)
          raise ArgumentError, "New FFN size must be larger" unless size > previous
          widths[layer] = size
          grown = ChoiceModel.new(old.with(model: { ffn_sizes: widths }))
          state = grown.state_dict
          Torch.no_grad do
            # New input rows random; new output columns zero. Old function is preserved.
            grown.encoder.blocks[layer].ffn.down.weight.zero!
            model.state_dict.each do |name, tensor|
              destination = state.fetch(name)
              tensor.shape.each_with_index { |length, dim| destination = destination.narrow(dim, 0, length) }
              destination.copy!(tensor.detach.cpu)
            end
          end
          grown.load_state_dict(state)
          grown
        end
      end
    end
  end
end
