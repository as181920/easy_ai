module EasyAI
  module NN
    # RoPE, adjacent feature pairs. Tensor shape [batch, heads, length, width].
    # Only self-attention uses this; separate query/state streams have no shared
    # position coordinate system for cross-attention.
    class RotaryPosition
      def self.call(tensor, offset: 0)
        width = tensor.shape.last
        raise ArgumentError, "Rotary head width must be even" unless width.even?
        positions = Torch.arange(tensor.shape[-2], dtype: :float32, device: tensor.device) + offset
        frequencies = Torch.exp(Torch.arange(0, width, 2, dtype: :float32, device: tensor.device) * (-Math.log(10_000.0) / width))
        angles = positions.unsqueeze(-1) * frequencies.unsqueeze(0)
        cosine, sine = Torch.cos(angles), Torch.sin(angles)
        pairs = tensor.reshape([*tensor.shape[0...-1], width / 2, 2])
        even, odd = pairs.select(-1, 0), pairs.select(-1, 1)
        Torch.stack([even * cosine - odd * sine, even * sine + odd * cosine], dim: -1).view(tensor.shape)
      end
    end
  end
end
