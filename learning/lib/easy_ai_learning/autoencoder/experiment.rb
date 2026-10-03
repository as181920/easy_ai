module EasyAILearning
  module Autoencoder
    module Experiment
      module_function

      def run(c)
        rows, valid = Course::Data.manifold(seed: c.seed), Course::Data.manifold(seed: c.seed + 1)
        c.artifacts.json("data", { train: rows, validation: valid })
        x, vx = c.tensor(rows), c.tensor(valid)
        { linear: [false, 0, false], bottleneck: [true, 0, false], sparse: [true, 0.05, false], denoising: [true, 0, true] }.each do |name, (nonlinear, sparsity, corrupt)|
          Torch.manual_seed(c.seed)
          model = c.model(Model.new(nonlinear: nonlinear))
          mask = Torch.rand_like(vx).ge(0.3)
          vin = corrupt ? Model.corrupt(vx, mask: mask) : vx
          c.train(name, model, validation: -> { Torch::NN::Functional.mse_loss(model.call(vin), vx) }) do
            input = corrupt ? Model.corrupt(x, mask: Torch.rand_like(x).ge(0.3)) : x
            model.objective(input, x, sparsity: sparsity)
          end
          c.results[name] = c.evaluate(model) do
            { validation_mse: Torch::NN::Functional.mse_loss(model.call(vin), vx).item,
              latent: Diagnostics::Stats.activation(model.encode(vx)) }
          end
          c.artifacts.json("#{name}-reconstruction", { input: vin.cpu.to_a, target: valid, output: c.evaluate(model) { model.call(vin).cpu.to_a } })
          c.save("#{name}-model", model, config: { input: 4, latent: 2, nonlinear: nonlinear })
        end
        pca = Foundations::Pca.new.fit(rows, dimensions: 2)
        c.results[:pca_validation_mse] = Foundations::Math.mse(pca.reconstruct(pca.transform(valid)).flatten, valid.flatten)
      end
    end
  end
end
