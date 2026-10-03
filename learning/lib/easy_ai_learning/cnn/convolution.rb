module EasyAILearning
  module CNN
    # Scalar cross-correlation reference for comparing the Torch Conv2d kernel.
    module Convolution
      module_function

      def output_size(input, kernel, stride: 1, padding: 0)
        raise ArgumentError, "Invalid convolution settings" unless stride > 0 && kernel > 0 && padding >= 0
        result = (input + 2 * padding - kernel) / stride + 1
        raise ArgumentError, "Kernel larger than input" unless result > 0
        result
      end

      def correlate(image, kernel, bias: 0.0, stride: 1, padding: 0)
        height = output_size(image.size, kernel.size, stride: stride, padding: padding)
        width = output_size(image.first.size, kernel.first.size, stride: stride, padding: padding)
        Array.new(height) do |row|
          Array.new(width) do |column|
            bias + kernel.each_index.sum do |i|
              kernel[i].each_index.sum do |j|
                r, c = row * stride + i - padding, column * stride + j - padding
                (r.between?(0, image.size - 1) && c.between?(0, image.first.size - 1) ? image[r][c] : 0) * kernel[i][j]
              end
            end
          end
        end
      end
    end
  end
end
