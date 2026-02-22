module EasyAI
  module Trainers
    class Trainer
      attr_reader :model, :batcher, :config, :optimizer, :device, :loss_history, :logger

      def initialize(model:, batcher:, config: EasyAI::Config.new.training, logger: EasyAI.logger)
        @model = model
        @batcher = batcher
        @config = config
        @device = config[:device]
        @optimizer = Torch::Optim::AdamW.new(model.parameters, lr: config[:lr], betas: [0.9, 0.95], weight_decay: config[:weight_decay])
        @loss_history = []
        @logger = logger
      end

      def train
        max_iters = config[:max_iters]
        log_interval = config[:log_interval]

        model.train
        max_iters.times do |iter|
          optimizer.zero_grad

          inputs, targets = batcher.next_batch
          inputs = inputs.to(device)
          targets = targets.to(device)

          logits = model.call(inputs)
          loss = Torch::NN::Functional.cross_entropy(
            logits.view([-1, logits.shape[-1]]),
            targets.view([-1])
          )
          loss.backward
          clip_gradients(model.parameters, config[:grad_clip]) if config[:grad_clip]
          optimizer.step

          # Detach loss to free computation graph memory
          loss_value = loss.detach.item

          # Explicitly clear tensors to help GC
          loss = nil
          logits = nil
          inputs = nil
          targets = nil

          # Periodically force Ruby GC to reclaim C++ tensor wrappers
          # that hold GPU memory. Ruby GC doesn't know about VRAM pressure,
          # so without this, stale wrappers accumulate and leak GPU memory.
          GC.start if (iter % 10).zero?

          loss_history << loss_value
          logger.debug { "[Trainer] iter=#{iter} loss=#{loss_value}" }
          yield(iter, loss_value) if block_given?

          if (iter % log_interval).zero?
            logger.info { "[Trainer] iter=#{iter} loss=#{loss_value}" }
          end
        end
      end

      private

        def clip_gradients(parameters, max_norm)
          return unless max_norm

          total_norm_sq = 0.0
          parameters.each do |param|
            grad = param.grad
            next unless grad

            total_norm_sq += grad.data.norm(2).item**2
          end

          total_norm = Math.sqrt(total_norm_sq)
          return if total_norm <= max_norm

          scale = max_norm / (total_norm + 1e-6)
          parameters.each do |param|
            grad = param.grad
            next unless grad

            grad.data.mul!(scale)
          end
        end
    end
  end
end
