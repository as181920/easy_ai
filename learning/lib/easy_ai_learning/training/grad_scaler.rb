module EasyAILearning
  module Training
    # Explicit loss scaling; no claim to provide a Torch autocast context.
    class GradScaler
      attr_reader :scale

      def initialize(scale: 128.0)
        raise ArgumentError, "Positive scale required" unless scale > 0
        @scale = scale
      end

      def backward(loss)
        (loss * scale).backward
      end

      def step(optimizer)
        parameters = optimizer.named.values
        finite = parameters.all? { |p| !p.grad || p.grad.numel.zero? || p.grad.isfinite.all.item }
        unless finite
          @scale /= 2
          optimizer.zero_grad
          return false
        end
        Torch.no_grad { parameters.each { |p| p.grad.div!(scale) if p.grad } }
        optimizer.step
        true
      end
    end
  end
end
