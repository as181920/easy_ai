require_relative "../test_helper"

class RLTest < Minitest::Test
  def test_chain_terminal_and_time_limit_are_different
    env = EasyAILearning::RL::Chain.new(size: 3, limit: 2)

    assert_equal [0, -0.02, false], env.transition(0, 0)
    assert_equal [2, 1.0, true], env.transition(1, 1)
    env.step(0)
    result = env.step(0)

    assert result[:truncated]
    refute result[:terminated]
    assert_raises(ArgumentError) { env.step(1) }
  end

  def test_terminal_q_update_and_bellman_solution
    q = [[0.0, 0.0], [10.0, 20.0]]
    EasyAILearning::RL::Tabular.update(q, state: 0, action: 1, reward: 1, next_state: 1, terminated: true, alpha: 0.5)

    assert_in_delta 0.5, q[0][1], 1e-12
    values = EasyAILearning::RL::Tabular.value_iteration(EasyAILearning::RL::Chain.new(size: 3), gamma: 0.9)

    assert_in_delta 1, values[1], 1e-12
    assert_in_delta 0.88, values[0], 1e-12
  end

  def test_returns_gae_and_dqn_targets_match_manual_values
    objectives = EasyAILearning::RL::Objectives

    assert_equal [2.0, 2.0], objectives.discounted_returns([1, 2], gamma: 0.5)
    assert_equal [1.5, 1.0], objectives.advantages([1, 1], [0, 0], [0, 0], [false, true], gamma: 0.5, gae_lambda: 1)
    target = objectives.dqn_targets(Torch.tensor([1.0, 1.0]), Torch.tensor([1.0, 0.0]), Torch.tensor([[10.0, 20.0], [2.0, 3.0]]), gamma: 0.5)

    assert_equal [1.0, 2.5], target.to_a
  end

  def test_ppo_clips_positive_and_negative_advantages
    logits = Torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    actions = Torch.tensor([0, 0], dtype: :int64)
    old_logp = Torch.tensor([Math.log(0.25), Math.log(0.75)])
    # Ratios: 2 and 2/3. Positive advantage clipped at 1.2; negative at 0.8.
    loss = EasyAILearning::RL::Objectives.ppo_loss(logits, actions, old_logp, Torch.tensor([1.0, -1.0]))

    assert_in_delta(-0.2, loss.item, 1e-6)
  end

  def test_policy_gradient_matches_analytic_softmax_gradient
    logits = Torch::NN::Parameter.new(Torch.zeros([1, 2]))
    loss = EasyAILearning::RL::Objectives.policy_loss(logits, Torch.tensor([0], dtype: :int64), Torch.tensor([2.0]))
    loss.backward

    assert_equal [[-1.0, 1.0]], logits.grad.to_a
  end

  def test_replay_capacity_and_sampling_never_mutates_source
    replay = EasyAILearning::RL::Replay.new(capacity: 2)
    original = [0, 1]
    replay.push(original)
    original[0] = 9

    assert_equal [0, 1], replay.entries.first
    replay.push([1, 2])
    replay.push([2, 3])

    assert_equal [[1, 2], [2, 3]], replay.entries
    assert_equal 2, replay.sample(2).uniq.size
  end
end

class TabularUpdatesTest < Minitest::Test
  def test_bandit_running_average_and_regret_on_deterministic_arm
    result = EasyAILearning::RL::Tabular.bandit(probabilities: [1.0], steps: 3, epsilon: 0)

    assert_equal [1.0], result[:values]
    assert_equal [3], result[:counts]
    assert_equal [[1, 0.0], [2, 0.0], [3, 0.0]], result[:regret]
  end

  def test_one_interaction_performs_exact_tabular_update
    environment = EasyAILearning::RL::Chain.new(size: 2, limit: 1)
    transitions = []
    environment.define_singleton_method(:step) do |action|
      result = super(action)
      transitions << [action, result]
      result
    end
    result = EasyAILearning::RL::Tabular.q_learning(environment, episodes: 1, seed: 8)
    action, observed = transitions.first

    assert_in_delta 0.2 * observed[:reward], result[:q][0][action], 1e-12
    assert_equal [observed[:reward]], result[:returns]
    assert_equal 1, transitions.size
  end
end

class ActorCriticTest < Minitest::Test
  def test_actor_and_critic_have_separate_gradient_paths
    model = EasyAILearning::RL::ActorCritic.new(states: 3)
    input = EasyAILearning::RL::Chain.one_hot([0, 2], size: 3)
    logits, values = model.call(input)
    logits.sum.backward

    assert_equal [2, 2], logits.shape
    assert_equal [2], values.shape
    assert model.policy.parameters.any? { |p| p.grad && p.grad.numel > 0 }
    assert model.value.parameters.all? { |p| p.grad.nil? || p.grad.numel.zero? }
  end
end
