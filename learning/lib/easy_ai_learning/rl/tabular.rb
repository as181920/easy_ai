module EasyAILearning
  module RL
    module Tabular
      module_function

      def bandit(probabilities: [0.2, 0.5, 0.8], steps: 100, epsilon: 0.2, seed: 1337)
        rng, values, counts = Random.new(seed), Array.new(probabilities.size, 0.0), Array.new(probabilities.size, 0)
        history, regret = [], 0.0
        steps.times do |step|
          action = rng.rand < epsilon ? rng.rand(values.size) : values.each_index.max_by { |i| values[i] }
          reward = rng.rand < probabilities[action] ? 1.0 : 0.0
          counts[action] += 1
          values[action] += (reward - values[action]) / counts[action]
          regret += probabilities.max - probabilities[action]
          history << [step + 1, regret]
        end
        { values: values, counts: counts, regret: history }
      end

      def value_iteration(environment, gamma: 0.95, steps: 100)
        values = Array.new(environment.size, 0.0)
        steps.times do
          values = values.each_index.map do |state|
            2.times.map do |action|
              next_state, reward, terminal = environment.transition(state, action)
              reward + (terminal ? 0 : gamma * values[next_state])
            end.max
          end
        end
        values
      end

      def update(q, state:, action:, reward:, next_state:, terminated:, alpha: 0.2, gamma: 0.95)
        target = reward + (terminated ? 0 : gamma * q.fetch(next_state).max)
        q[state][action] += alpha * (target - q[state][action])
      end

      def q_learning(environment, episodes: 100, seed: 1337)
        q, rng, returns = Array.new(environment.size) { [0.0, 0.0] }, Random.new(seed), []
        episodes.times do
          state, total = environment.reset, 0.0
          loop do
            action = rng.rand < 0.3 ? rng.rand(2) : q[state].each_index.max_by { |i| q[state][i] }
            result = environment.step(action)
            update(q, state: state, action: action, reward: result[:reward], next_state: result[:state], terminated: result[:terminated])
            total += result[:reward]
            state = result[:state]
            break if result[:terminated] || result[:truncated]
          end
          returns << total
        end
        { q: q, returns: returns }
      end
    end
  end
end
