module EasyAILearning
  module Transfer
    module Distillation
      module_function

      def loss(student_logits, teacher_logits, targets, temperature: 2.0, soft_weight: 0.5)
        raise ArgumentError, "Invalid distillation settings" unless temperature > 0 && soft_weight.between?(0, 1)
        target_probability = Torch::NN::Functional.softmax(teacher_logits.detach / temperature, dim: -1)
        student_log_probability = (student_logits / temperature).log_softmax(-1)
        soft = -(target_probability * student_log_probability).sum(-1).mean * temperature**2
        hard = Training::Math.masked_cross_entropy(student_logits, targets)
        soft_weight * soft + (1 - soft_weight) * hard
      end
    end
  end
end
