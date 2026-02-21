module EasyAI
  class Config
    DEFAULTS = {
      model: {
        vocab_size: 50257,
        block_size: 128,
        n_layer: 4,
        n_head: 4,
        n_embd: 256,
        dropout: 0.1
      },
      training: {
        batch_size: 16,
        lr: 3e-4,
        weight_decay: 0.1,
        max_iters: 200,
        log_interval: 10,
        device: Torch.device(Torch::CUDA.available? ? "cuda" : "cpu"),
        grad_clip: 1.0,
        seed: 1337
      }
    }.freeze

    attr_reader :options

    def initialize(overrides = {})
      @options = DEFAULTS.deep_merge(overrides.deep_symbolize_keys)
    end

    def model
      options[:model]
    end

    def training
      options[:training]
    end

    def [](key)
      options[key]
    end
  end
end
