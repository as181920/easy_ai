module EasyAILearning
  module RL
    module Objectives
      module_function

      def discounted_returns(rewards, gamma: 0.95)
        total = 0.0
        rewards.reverse.map { |reward| total = reward + gamma * total }.reverse
      end

      def dqn_targets(rewards, terminated, next_values, gamma: 0.95)
        rewards + gamma * (1 - terminated.to(dtype: rewards.dtype)) * next_values.detach.max(1).first
      end

      def policy_loss(logits, actions, advantages)
        logp = logits.log_softmax(-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        -(logp * advantages.detach).mean
      end

      def ppo_loss(logits, actions, old_logp, advantages, clip: 0.2)
        logp = logits.log_softmax(-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = (logp - old_logp.detach).exp
        clipped = ratio.clamp(1 - clip, 1 + clip)
        -Torch.minimum(ratio * advantages.detach, clipped * advantages.detach).mean
      end

      def advantages(rewards, values, next_values, terminated, gamma: 0.95, gae_lambda: 0.95)
        raise ArgumentError, "Trajectory lengths differ" unless [rewards, values, next_values, terminated].map(&:size).uniq.size == 1
        last = 0.0
        rewards.each_index.to_a.reverse.map do |i|
          continuation = terminated[i] ? 0.0 : 1.0
          delta = rewards[i] + gamma * continuation * next_values[i] - values[i]
          last = delta + gamma * gae_lambda * continuation * last
        end.reverse
      end
    end
  end
end
