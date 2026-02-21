module EasyAI
  module Utils
    module TensorOps
      module_function

      def causal_mask(size, device: Torch.device("cpu"))
        Torch.tril(Torch.ones([size, size], dtype: :bool, device: device))
      end

      def gelu(x)
        Torch::NN::Functional.gelu(x)
      end
    end
  end
end
