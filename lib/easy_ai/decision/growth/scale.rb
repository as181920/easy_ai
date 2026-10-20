module EasyAI
  module Decision
    module Growth
      # Enlarge depth and FFNs while preserving the parent's initial function.
      class Scale
        def self.apply(model, layers:, ffn_size:, seed:)
          old = model.config
          widths = old[:model]["ffn_sizes"] || Array.new(old[:model]["encoder_layers"], old[:model]["ffn_size"])
          unless layers.is_a?(Integer) && ffn_size.is_a?(Integer) && layers >= widths.size && widths.all? { |width| ffn_size >= width }
            raise ArgumentError, "Scaling cannot remove layers or FFN units"
          end
          Torch.manual_seed(seed)
          config = old.with(model: { encoder_layers: layers, ffn_size: ffn_size, ffn_sizes: Array.new(layers, ffn_size) })
          grown = ChoiceModel.new(config)
          state = grown.state_dict
          Torch.no_grad do
            state.each { |name, tensor| tensor.zero! if name.end_with?(".ffn.down.weight") }
            model.state_dict.each do |name, tensor|
              destination = state.fetch(name)
              unless tensor.shape.size == destination.shape.size && tensor.shape.zip(destination.shape).all? { |a, b| a <= b }
                raise ArgumentError, "Incompatible scaling tensor: #{name}"
              end
              tensor.shape.each_with_index { |length, dim| destination = destination.narrow(dim, 0, length) }
              destination.copy!(tensor.detach.cpu)
            end
          end
          grown.load_state_dict(state)
          (widths.size...layers).each { |index| grown.encoder.blocks[index].identity! }
          grown
        end
      end
    end
  end
end
