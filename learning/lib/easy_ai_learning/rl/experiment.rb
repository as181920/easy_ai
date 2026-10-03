module EasyAILearning
  module RL
    module Experiment
      module_function

      def run(c)
        environment = Chain.new
        bandit = Tabular.bandit(steps: c.steps * 4, seed: c.seed)
        c.artifacts.json("bandit", bandit)
        c.artifacts.plot("bandit-regret", { expected_regret: bandit[:regret] })
        c.results[:bandit] = bandit.slice(:values, :counts)
        c.results[:exact_values] = Tabular.value_iteration(environment)
        tabular = Tabular.q_learning(environment, episodes: c.steps, seed: c.seed)
        c.artifacts.json("q-learning", tabular)
        c.artifacts.plot("q-learning-return", { episode_return: tabular[:returns].each_with_index.map { |r, i| [i + 1, r] } })
        c.results[:q_learning] = evaluate(environment) { |state| tabular[:q][state].each_index.max_by { |i| tabular[:q][state][i] } }
        c.artifacts.json("environment", { states: environment.size, actions: %w[left right], max_steps: environment.limit,
          transitions: environment.size.times.flat_map { |s| 2.times.map { |a| [s, a, environment.transition(s, a)] } } })
        train_dqn(c, environment)
        %i[reinforce actor_critic ppo].each { |kind| train_policy(c, environment, kind) }
      end

      def evaluate(environment)
        episodes = (0...(environment.size - 1)).map do |start|
          state, total, path = environment.reset(start: start), 0.0, [start]
          loop do
            result = environment.step(yield(state))
            total += result[:reward]
            state = result[:state]
            path << state
            break if result[:terminated] || result[:truncated]
          end
          { start: start, return: total, success: state == environment.size - 1, path: path }
        end
        { mean_return: episodes.sum { |r| r[:return] } / episodes.size, success_rate: episodes.count { |r| r[:success] }.to_f / episodes.size, episodes: episodes }
      end

      def train_dqn(c, env)
        Torch.manual_seed(c.seed)
        model = c.model(BasicNN::Mlp.new(input: env.size, hidden: 12, output: 2))
        target = c.model(BasicNN::Mlp.new(input: env.size, hidden: 12, output: 2))
        target.load_state_dict(model.state_dict)
        target.eval
        optimizer = Training::Optimizer.new(model.named_parameters)
        replay, rng, history = Replay.new(seed: c.seed), Random.new(c.seed), []
        c.steps.times do |episode|
          state, total = env.reset, 0.0
          loop do
            action = rng.rand < 0.3 ? rng.rand(2) : Torch.no_grad { model.call(Chain.one_hot([state], size: env.size, device: c.device)).argmax(-1).item }
            result = env.step(action)
            replay.push([state, action, result[:reward], result[:state], result[:terminated]])
            total += result[:reward]
            state = result[:state]
            break if result[:terminated] || result[:truncated]
          end
          samples = replay.sample([16, replay.entries.size].min)
          states, actions, rewards, next_states, terminated = samples.transpose
          optimizer.zero_grad
          prediction = model.call(Chain.one_hot(states, size: env.size, device: c.device)).gather(1, c.tensor(actions, integer: true).unsqueeze(1)).squeeze(1)
          next_values = Torch.no_grad { target.call(Chain.one_hot(next_states, size: env.size, device: c.device)) }
          expected = Objectives.dqn_targets(c.tensor(rewards), c.tensor(terminated.map { |v| v ? 1.0 : 0.0 }), next_values)
          loss = Torch::NN::Functional.mse_loss(prediction, expected)
          loss.backward
          Training::Math.clipped_gradients(model.parameters, 5)
          optimizer.step
          target.load_state_dict(model.state_dict) if (episode + 1) % 5 == 0
          history << [episode + 1, total, loss.item]
        end
        model.eval
        c.results[:dqn] = evaluate(env) { |state| Torch.no_grad { model.call(Chain.one_hot([state], size: env.size, device: c.device)).argmax(-1).item } }
        c.artifacts.json("dqn-history", history)
        c.artifacts.plot("dqn-return", { episode_return: history.map { |step, total, _| [step, total] } })
        c.artifacts.json("replay", replay.entries)
        c.save("dqn-model", model, config: { input: 5, hidden: 12, output: 2 })
      end

      def rollout(c, env, model)
        state, states, actions, rewards, terminated, next_states = env.reset, [], [], [], [], []
        loop do
          logits, = Torch.no_grad { model.call(Chain.one_hot([state], size: env.size, device: c.device)) }
          action = Torch.multinomial(Torch::NN::Functional.softmax(logits, dim: -1), num_samples: 1).item
          result = env.step(action)
          states << state
          actions << action
          rewards << result[:reward]
          terminated << result[:terminated]
          next_states << result[:state]
          state = result[:state]
          break if result[:terminated] || result[:truncated]
        end
        { states: states, actions: actions, rewards: rewards, terminated: terminated, next_states: next_states }
      end

      def train_policy(c, env, kind)
        Torch.manual_seed(c.seed)
        model = c.model(ActorCritic.new)
        optimizer = Training::Optimizer.new(model.named_parameters, lr: 0.01)
        history = []
        c.steps.times do |episode|
          trajectory = rollout(c, env, model)
          inputs = Chain.one_hot(trajectory[:states], size: env.size, device: c.device)
          actions = c.tensor(trajectory[:actions], integer: true)
          old_logits, values = Torch.no_grad { model.call(inputs) }
          next_values = Torch.no_grad { model.call(Chain.one_hot(trajectory[:next_states], size: env.size, device: c.device)).last }
          advantage = kind == :reinforce ? Objectives.discounted_returns(trajectory[:rewards]) : Objectives.advantages(
            trajectory[:rewards], values.cpu.to_a, next_values.cpu.to_a, trajectory[:terminated])
          adv = c.tensor(advantage)
          value_target = values.detach + adv
          old_logp = old_logits.log_softmax(-1).gather(1, actions.unsqueeze(1)).squeeze(1).detach
          loss_value = 0.0
          (kind == :ppo ? 4 : 1).times do
            optimizer.zero_grad
            logits, value = model.call(inputs)
            loss = kind == :ppo ? Objectives.ppo_loss(logits, actions, old_logp, adv) : Objectives.policy_loss(logits, actions, adv)
            loss = loss + 0.5 * Torch::NN::Functional.mse_loss(value, value_target) unless kind == :reinforce
            loss.backward
            Training::Math.clipped_gradients(model.parameters, 5)
            optimizer.step
            loss_value = loss.item
          end
          history << [episode + 1, trajectory[:rewards].sum, loss_value]
        end
        model.eval
        c.results[kind] = evaluate(env) { |state| Torch.no_grad { model.call(Chain.one_hot([state], size: env.size, device: c.device)).first.argmax(-1).item } }
        c.artifacts.json("#{kind}-history", history)
        c.artifacts.plot("#{kind}-return", { episode_return: history.map { |step, total, _| [step, total] } })
        c.save("#{kind}-model", model, config: { states: 5 })
      end
    end
  end
end
