module EasyAILearning
  module Generative
    class Diffusion < Torch::NN::Module
      attr_reader :betas, :alpha_bar, :count, :network

      def initialize(count: 12, beta_start: 0.02, beta_end: 0.15)
        super()
        raise ArgumentError, "Invalid diffusion schedule" unless count > 1 && beta_start > 0 && beta_end >= beta_start && beta_end < 1
        @count = count
        @betas = Array.new(count) { |i| beta_start + (beta_end - beta_start) * i / (count - 1).to_f }
        product = 1.0
        @alpha_bar = betas.map { |beta| product *= 1 - beta }
        @network = BasicNN::Mlp.new(input: 3, hidden: 24, output: 2, activation: :tanh)
        register_buffer("cumulative_alpha", Torch.tensor(alpha_bar, dtype: :float32))
      end

      def q_sample(x, time, noise:)
        cumulative = state_dict.fetch("cumulative_alpha").index_select(0, time).unsqueeze(1)
        cumulative.sqrt * x + (1 - cumulative).sqrt * noise
      end

      def forward(x, time)
        t = time.to(dtype: x.dtype).unsqueeze(1) / (count - 1)
        network.call(Torch.cat([x, t], dim: 1))
      end

      def objective(x, time:, noise:)
        Torch::NN::Functional.mse_loss(forward(q_sample(x, time, noise: noise), time), noise)
      end

      def reverse_step(x, time:, predicted_noise:, noise:)
        beta, cumulative = betas.fetch(time), alpha_bar.fetch(time)
        previous = time.zero? ? 1.0 : alpha_bar.fetch(time - 1)
        mean = (x - beta / ::Math.sqrt(1 - cumulative) * predicted_noise) / ::Math.sqrt(1 - beta)
        variance = beta * (1 - previous) / (1 - cumulative)
        mean + ::Math.sqrt(variance) * noise
      end

      def sample(size: 64, device: Torch.device("cpu"), initial: nil)
        Torch.no_grad do
          x = initial || Torch.randn([size, 2], device: device)
          (count - 1).downto(0) do |step|
            time = Torch.full([x.shape[0]], step, dtype: :int64, device: device)
            x = reverse_step(x, time: step, predicted_noise: forward(x, time), noise: step.zero? ? Torch.zeros_like(x) : Torch.randn_like(x))
          end
          x
        end
      end
    end
  end
end
