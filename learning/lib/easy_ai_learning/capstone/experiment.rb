module EasyAILearning
  module Capstone
    module Experiment
      module_function

      def run(c)
        rows, labels = Course::Data.images(seed: c.seed)
        valid, vlabels = Course::Data.images(seed: c.seed + 1)
        test, tlabels = Course::Data.images(seed: c.seed + 2)
        c.artifacts.json("data", { train: [rows, labels], validation: [valid, vlabels], test: [test, tlabels] })
        x, y, vx, vy, tx, ty = c.tensor(rows), c.tensor(labels, integer: true), c.tensor(valid), c.tensor(vlabels, integer: true), c.tensor(test), c.tensor(tlabels, integer: true)
        seeds = [c.seed, c.seed + 10, c.seed + 20]
        panel = []
        seeds.each do |seed|
          %i[mlp cnn resnet].each do |kind|
            Torch.manual_seed(seed)
            raw, config = case kind
                          when :mlp then [BasicNN::Mlp.new(input: 64, hidden: 12), { input: 64, hidden: 12, output: 2 }]
                          when :cnn then [CNN::Model.new, {}]
                          when :resnet then [Resnet::Model.new(depth: 2), { depth: 2 }]
                          end
            model = c.model(raw)
            train_x, valid_x, test_x = kind == :mlp ? [x, vx, tx].map { |t| t.reshape([64, 64]) } : [x, vx, tx]
            name = "#{kind}-#{seed}"
            c.train(name, model, validation: -> { Training::Math.masked_cross_entropy(model.call(valid_x), vy) }) do
              Training::Math.masked_cross_entropy(model.call(train_x), y)
            end
            metrics = c.evaluate(model) do
              { validation_accuracy: c.accuracy(model.call(valid_x), vy), test_accuracy: c.accuracy(model.call(test_x), ty) }
            end
            c.save("#{name}-model", model, config: config)
            restored = c.model(kind == :mlp ? BasicNN::Mlp.new(**config) : kind == :cnn ? CNN::Model.new : Resnet::Model.new(**config))
            Course::Artifacts.load_model(File.join(c.artifacts.directory, "#{name}-model.json"), restored)
            difference = c.evaluate(model) { (model.call(test_x) - restored.call(test_x)).abs.max.item }
            panel << metrics.merge(model: kind, seed: seed, reload_max_difference: difference, parameters: model.parameters.sum(&:numel))
          end
        end
        c.artifacts.json("panel", panel)
        c.results[:panel] = panel
        c.results[:summary] = panel.group_by { |row| row[:model] }.transform_values do |rows|
          values = rows.map { |r| r[:test_accuracy] }
          mean = values.sum / values.size
          { mean: mean, std: ::Math.sqrt(values.sum { |v| (v - mean)**2 } / values.size), min: values.min, max: values.max }
        end
        c.results[:scope] = "Preselected architectures, three initialization seeds, independently generated small image splits. This is a toy benchmark, not external-data evidence."
      end
    end
  end
end
