# 15 · Reward-driven learning: Bandit → PPO

A minimal teaching experiment is implemented and runnable. Prerequisites: 01–03; GPT is not required. Next chapter: [16_capstone](../16_capstone/README.md).

Start with a three-arm Bernoulli bandit to understand epsilon-greedy exploration, sample means, and expected regret. Then learn left/right actions in a five-state chain. Reaching the rightmost goal rewards 1; other transitions reward -0.02, with a maximum of 12 steps. The environment is entirely local and does not require GPT or the production Decision module.

| Method | Implementation and reuse |
| --- | --- |
| Exact value iteration | Known transitions/rewards; Bellman reference |
| Tabular Q-learning | Update Q through interaction; no bootstrap at terminal states |
| DQN | MLP Q from 01; bounded replay, target synchronization every five episodes, detached targets |
| REINFORCE | Policy MLP from 01; Monte Carlo return × log-probability; no trained critic |
| Actor-critic | Policy + value MLP; GAE advantages and value targets |
| PPO | Four updates on the same rollout; detached old logp, ratio clipping, and value loss |

```text
Q ← Q+α[r+γ(1-terminal)max Q(next)-Q]
L_policy=-mean(logπ(a|s)*advantage)
L_PPO=-mean(min(ratio*A,clip(ratio,1-ε,1+ε)*A))
```

Termination and time-limit truncation are separate. Termination removes the next-state value from the target; truncation can bootstrap under a continuing-task interpretation. GAE is computed backward within one rollout without mixing later episodes into its endpoint. REINFORCE uses sampled returns without an untrained value bootstrap. Its network includes a value branch for a shared interface, but that branch does not participate in the objective or update.

Evaluation disables updates and runs the greedy policy from each of four nonterminal starting states, reporting success rate, return, and paths. The environment is deterministic, so evaluation seeds cannot create real environmental variation; training action sampling remains random. The default uses only one training seed. Apply the multi-seed method from 16 for further comparisons, and do not treat critic loss as task success.

JSON includes environment transitions, replay, and histories; neural models support independent inference. This chapter does not provide a CLI for restoring full RL state mid-episode, so model JSON should not be described as complete RL resume state. Source: [PPO](https://arxiv.org/abs/1707.06347).

## Data, training, and independent inference

Run commands from the repository root and install dependencies with `bundle install`. Training and model inference default to `auto`: prefer CUDA and fall back to CPU when unavailable. You can also select `--device cpu` explicitly. These experiments use small synthetic datasets and require no model downloads. Limiting threads reduces CPU overhead for tiny tensors:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/15_rl/data.rb
bundle exec ruby learning/15_rl/train.rb --steps 60
bundle exec ruby learning/15_rl/predict.rb
bundle exec rake test:learning
```

Use `--seed` to change the random seed and `--output` to separate experiment directories. Training also accepts `--device cpu/cuda/auto`. The default output is `runs/learning/15_rl/default/`; rerunning overwrites artifacts with the same names. `data.rb` exports samples from the data recipe for inspection. The training entry point calls the generators directly and does not depend on that JSON file. Experiment-specific shifts and masks are documented in the experiment source and the actual `data.json`.

For inference, `--model PATH` selects a saved model. `--input PATH` accepts a JSON file containing `{"input": ...}`; the default is a small example with the required shape. Seq2seq/EncoderDecoder perform free-running generation, GPT uses top-1 generation, and other models return scores or reconstructions. RL actor-critic models return policy logits and a value estimate.

## Results and correctness

The [recorded run](results.json) includes the seed, step count, environment, and metrics. The figure below comes from that run; it is not a test acceptance threshold.

![Chapter experiment results](images/ppo-return.svg)

[Experiment code](../lib/easy_ai_learning/rl/experiment.rb) connects the steps; shared data generators are in [course/data.rb](../lib/easy_ai_learning/course/data.rb). Core checks are in the [tests](../test/course/rl_test.rb), with additional gradient comparisons in [derivatives_test.rb](../test/course/derivatives_test.rb). Tests check deterministic formulas, shapes, masks, gradients, state, and parameter updates. They do not train toward a required accuracy or weight distribution.

Artifacts include the actual data, JSON inference state, history, diagnostics, and SVG figures. Full local parameters and histories remain under the ignored `runs/` directory; the repository contains only compact result summaries and figures. The non-neural experiments in 00/07 and the interactive training in 15 use task-specific records rather than forcing every result into a classification-loss format.
