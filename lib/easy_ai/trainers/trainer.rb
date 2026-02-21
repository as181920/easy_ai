module EasyAI
  module Trainers
    class Trainer
      attr_reader :model, :batcher, :config, :optimizer, :device

      def initialize(model:, batcher:, config: EasyAI::Config.new.training)
        @model = model
        @batcher = batcher
        @config = config
        @device = config[:device]
        @optimizer = Torch::Optim::AdamW.new(model.parameters, lr: config[:lr], betas: [0.9, 0.95], weight_decay: config[:weight_decay])
      end

      def train
        max_iters = config[:max_iters]
        log_interval = config[:log_interval]

        model.train
        max_iters.times do |iter|
          inputs, targets = batcher.next_batch
          inputs = inputs.to(device)
          targets = targets.to(device)

          optimizer.zero_grad
          logits = model.call(inputs)
          loss = Torch::NN::Functional.cross_entropy(
            logits.view([-1, logits.shape[-1]]),
            targets.view([-1])
          )
          loss.backward
          Torch::NN::Utils.clip_grad_norm!(model.parameters, config[:grad_clip]) if config[:grad_clip]
          optimizer.step

          if (iter % log_interval).zero?
            puts "iter=#{iter} loss=#{loss.item}"
          end
        end
      end
    end
  end
end
