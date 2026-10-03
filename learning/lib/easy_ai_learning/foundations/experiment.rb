module EasyAILearning
  module Foundations
    module Experiment
      module_function

      def run(c)
        x, y = Course::Data.regression(seed: c.seed)
        xv, yv = Course::Data.regression(seed: c.seed + 1)
        linear = Linear.new(features: 1).fit(x, y.flatten, steps: c.steps)
        c.artifacts.json("data", { train: [x, y], validation: [xv, yv] })
        c.artifacts.json("linear-model", { weights: linear.weights, bias: linear.bias })
        c.artifacts.plot("linear-loss", { mse: linear.history.each_with_index.map { |v, i| [i + 1, v] } })
        c.results[:linear] = { train_mse: linear.loss(x, y.flatten), validation_mse: linear.loss(xv, yv.flatten),
          manual_derivative: 6.0, numerical_derivative: Math.finite_difference(3) { |v| v * v } }
        cx, cy = Course::Data.classification(seed: c.seed)
        vx, vy = Course::Data.classification(seed: c.seed + 1)
        logistic = Linear.new(features: 2, logistic: true).fit(cx, cy, steps: c.steps)
        c.artifacts.json("logistic-model", { weights: logistic.weights, bias: logistic.bias })
        c.results[:classification] = { logistic_accuracy: Math.confusion(vx.map { |r| logistic.predict(r) >= 0.5 ? 1 : 0 }, vy)[:accuracy] }
        { tree: Tree.new.fit(cx, cy), forest: Ensemble.new.fit(cx, cy), boosting: Ensemble.new.fit(cx, cy, kind: :boosting) }.each do |name, model|
          c.results[:classification][name] = Math.confusion(vx.map { |r| model.predict(r) >= 0.5 ? 1 : 0 }, vy)[:accuracy]
        end
        rows = Course::Data.manifold(seed: c.seed)
        pca = Pca.new.fit(rows, dimensions: 2)
        reconstructed = pca.reconstruct(pca.transform(rows))
        c.results[:pca] = { eigenvalues: pca.eigenvalues, reconstruction_mse: Math.mse(reconstructed.flatten, rows.flatten) }
        c.artifacts.json("pca-model", { mean: pca.mean, components: pca.components })
        cluster_data = Course::Data.mixture(seed: c.seed)
        clusters = KMeans.new.fit(cluster_data, seed: c.seed)
        c.results[:kmeans] = { centers: clusters.centers }
        c.artifacts.json("clusters", { data: cluster_data, labels: cluster_data.map { |r| clusters.predict(r) } })
        c.results[:probability] = { softmax: Math.softmax([1000, 1001]), cross_entropy: Math.cross_entropy([1000, 1001], 1) }
      end
    end
  end
end
