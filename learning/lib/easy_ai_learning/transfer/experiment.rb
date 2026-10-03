module EasyAILearning
  module Transfer
    module Experiment
      module_function

      def run(c)
        source, sy = Course::Data.classification(seed: c.seed)
        target, ty = Course::Data.classification(count: 32, seed: c.seed + 1)
        valid, vy = Course::Data.classification(seed: c.seed + 2)
        # Domain shift in observed coordinates; class remains based on underlying coordinates.
        target = target.map { |a, b| [a + 0.3, b - 0.2] }
        valid = valid.map { |a, b| [a + 0.3, b - 0.2] }
        c.artifacts.json("data", { source: [source, sy], target_train: [target, ty], target_validation: [valid, vy] })
        sx, sy, tx, ty, vx, vy = c.tensor(source), c.tensor(sy, integer: true), c.tensor(target), c.tensor(ty, integer: true), c.tensor(valid), c.tensor(vy, integer: true)
        teacher = c.model(BasicNN::Mlp.new)
        c.train("pretrain", teacher) { Training::Math.masked_cross_entropy(teacher.call(sx), sy) }
        c.save("pretrained-model", teacher, config: { input: 2, hidden: 12, output: 2 })
        variants = %i[scratch frozen partial full lora]
        variants.each do |name|
          model = c.model(name == :lora ? AdaptedMlp.new : BasicNN::Mlp.new)
          unless name == :scratch
            if name == :lora
              model.hidden.load_state_dict(teacher.hidden.state_dict)
              model.output.base.load_state_dict(teacher.output.state_dict)
            else
              model.load_state_dict(teacher.state_dict)
              model.hidden.parameters.each { |p| p.requires_grad = false } if %i[frozen partial].include?(name)
              # Partial fine-tuning updates the hidden bias as well as the head.
              model.hidden.bias.requires_grad = true if name == :partial
            end
          end
          before = Diagnostics::Stats.snapshot(model)
          c.train(name, model, lr: 0.01, validation: -> { Training::Math.masked_cross_entropy(model.call(vx), vy) }) do
            Training::Math.masked_cross_entropy(model.call(tx), ty)
          end
          c.results[name] = c.evaluate(model) { { accuracy: c.accuracy(model.call(vx), vy), trainable_parameters: model.parameters.select(&:requires_grad).sum(&:numel),
            updates: Diagnostics::Stats.updates(model, before) } }
          c.save("#{name}-model", model, config: { input: 2, hidden: 12, output: 2 })
        end
        student = c.model(BasicNN::Mlp.new(hidden: 4))
        teacher_logits = c.evaluate(teacher) { teacher.call(tx).detach }
        c.train("distilled", student, validation: -> { Training::Math.masked_cross_entropy(student.call(vx), vy) }) do
          Distillation.loss(student.call(tx), teacher_logits, ty)
        end
        c.results[:distilled] = c.evaluate(student) { { accuracy: c.accuracy(student.call(vx), vy), parameters: student.parameters.sum(&:numel) } }
        c.save("distilled-model", student, config: { input: 2, hidden: 4, output: 2 })
      end
    end
  end
end
