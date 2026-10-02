# Reinforcement learning

Status: **planned learning stage**. This directory defines the progression; no executable RL trainer is implemented yet. Teaching implementations will use Ruby and Torch.rb under `EasyAILearning::RL`, separate from production Decision training.

## What changes after supervised learning?

Supervised learning compares a prediction with a supplied target. RL learns a policy from interaction and rewards, with the objective of maximizing expected return. A policy can be a small MLP, an RNN or a Transformer/GPT; RL is not another network layer or an obligatory next step for every task.

```text
                      action a_t
        policy --------------------------> environment
          ^                                     |
          |        next observation, reward     |
          +-------------------------------------+

        collect transitions/episodes
                    |
        estimate values, returns or advantages
                    |
        update value function and/or policy
                    |
        evaluate task success with learning disabled
```

For sequential decisions, the usual formalism is a **Markov decision process (MDP)**: states, actions, transitions, rewards and a discount factor. A Markov chain describes transitions without an agent's action choice; an MDP adds decisions and rewards. The Markov property means the current state contains the information needed to determine the next-state distribution given an action. When observations omit that information, history or recurrent state may be needed.

```text
s_(t+1) ~ P(. | s_t, a_t)
r_t     = R(s_t, a_t, s_(t+1))
G_t     = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
```

RL is not restricted to multi-step tasks: a bandit provides a useful one-step starting point. Multi-step examples introduce delayed rewards and credit assignment.

## Planned lessons

| Step | Small task | Main learning objective |
| --- | --- | --- |
| 1. Bandit | Choose among a few reward-producing arms | Exploration, exploitation, expected reward and regret |
| 2. MDP and exact reference | Tiny grid/chain with known transitions | States/actions, termination, discounted return and Bellman updates |
| 3. Tabular Q-learning | Learn the same environment through interaction | Temporal-difference targets and comparison with the exact solution |
| 4. Neural Q-learning / DQN | Replace the table with a small MLP | Torch gradients, replay buffer and a target network |
| 5. REINFORCE | Train action probabilities in a small environment | Log-probability gradients, returns and a baseline |
| 6. Actor-critic, then PPO | Reuse a verified environment and policy | Advantages, a learned value baseline and constrained policy updates |

The exact-reference solver knows the environment; the learning agents must learn from sampled interaction. Keep that distinction visible. Add each runnable lesson only after the preceding behavior is understood and verified.

## Implementation and reporting conventions

- Ruby owns the environment and training loop. Torch.rb owns numerical tables/tensors, neural forward passes and gradients. Prefer CUDA with CPU fallback; tiny lessons may run faster on CPU, so GPU use is a learning choice rather than a speed claim.
- Keep environments, agents, rollout/replay storage and reporting separate. Add shared code under `learning/lib/easy_ai_learning/rl/` and tests under `learning/test/rl/` when implementation begins.
- Save weights, optimizer state, configuration, seeds and learning history under ignored `runs/learning/rl/<lesson>/<run>/`. Persist any environment/replay/RNG state required for genuine training continuation; inference-only weights are not a complete resume artifact.
- Report return, success rate, episode length and exploration alongside any loss. A descending critic loss does not establish a good policy. Evaluate with updates disabled and a documented action-selection policy, over separate evaluation seeds and multiple training seeds.
- Begin with `unicode_plot`: learning return/regret, a grid of learned values/actions and sample trajectories. Use gnuplot for a reviewed README figure when necessary.

## Relation to GPT and Decision

GPT teaches next-token prediction; RL teaches reward-driven behavior. Language-model RL post-training is a later application, after the reward, rollout, value and policy-gradient machinery is understood. A standalone toy environment is the first implementation target here.

The current Decision model learns candidate judgments from labelled examples. RL is not a replacement for its missing semantic supervision. Future multi-step planning or workflow agents may use Decision scores as inputs, but neither production RL nor business integration is introduced by this roadmap addition.

Reference: [Spinning Up: key concepts in reinforcement learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).
