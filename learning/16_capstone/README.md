# 16 · Capstone: from data to independent inference

A minimal teaching experiment is implemented and runnable. Prerequisites: 00–03, 05–06.

This chapter completes the vision workflow by reusing training/state handling from 02, diagnostics from 03, CNN from 05, and ResNet from 06. Compare MLP, CNN, and a two-block ResNet on the same 64/64/64 train/validation/test images using initialization seeds 1337/1347/1357.

Keep architectures and hyperparameters fixed rather than choosing models by test results. Each run records validation accuracy, final test accuracy, parameter count, curves, and per-layer diagnostics. After saving, create a fresh model of the same architecture, load it, compare inference, and report the maximum reload difference. Summarize test mean/std/min/max across runs. The three seeds produce actual parameter changes; training accuracy has no hard acceptance threshold.

```ruby
model.load_state_dict(saved_state)
# Compare logits before/after loading on the same data in eval/no_grad mode
reload_difference = (original_logits - restored_logits).abs.max.item
```

The generator produces easily separable horizontal/vertical lines. All models reaching 100% only shows that this task is too easy; it is not evidence of real visual robustness or model rankings. Next, add noise, vary lengths/positions, or use harder data. Design experiments on validation data, then perform a final independent test.

Other curriculum branches can use the same delivery standard: RNN/GPT sequence prediction, PCA/AE/VAE representation and generation, or Bandit/Q-learning/MLP policies. Their earlier chapters already provide small experiments. This chapter's default runnable capstone is the vision comparison; it does not claim to run multi-seed projects for every branch.

Completion criteria: data sources and splits, baselines, shapes/parameter counts, seeds/budgets, task metrics, failure evidence, ablations, and independent inference loading. Deterministic tests still establish core logic correctness; experiment results support learning.

## Data, training, and independent inference

Run commands from the repository root and install dependencies with `bundle install`. Training and model inference default to `auto`: prefer CUDA and fall back to CPU when unavailable. You can also select `--device cpu` explicitly. These experiments use small synthetic datasets and require no model downloads. Limiting threads reduces CPU overhead for tiny tensors:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/16_capstone/data.rb
bundle exec ruby learning/16_capstone/train.rb --steps 60
bundle exec ruby learning/16_capstone/predict.rb
bundle exec rake test:learning
```

Use `--seed` to change the random seed and `--output` to separate experiment directories. Training also accepts `--device cpu/cuda/auto`. The default output is `runs/learning/16_capstone/default/`; rerunning overwrites artifacts with the same names. `data.rb` exports samples from the data recipe for inspection. The training entry point calls the generators directly and does not depend on that JSON file. Experiment-specific shifts and masks are documented in the experiment source and the actual `data.json`.

For inference, `--model PATH` selects a saved model. `--input PATH` accepts a JSON file containing `{"input": ...}`; the default is a small example with the required shape. Seq2seq/EncoderDecoder perform free-running generation, GPT uses top-1 generation, and other models return scores or reconstructions. RL actor-critic models return policy logits and a value estimate.

## Results and correctness

The [recorded run](results.json) includes the seed, step count, environment, and metrics. The figure below comes from that run; it is not a test acceptance threshold.

![Chapter experiment results](images/cnn-1337-loss.svg)

[Experiment code](../lib/easy_ai_learning/capstone/experiment.rb) connects the steps; shared data generators are in [course/data.rb](../lib/easy_ai_learning/course/data.rb). Core checks are in the [tests](../test/course/training_test.rb), with additional gradient comparisons in [derivatives_test.rb](../test/course/derivatives_test.rb). Tests check deterministic formulas, shapes, masks, gradients, state, and parameter updates. They do not train toward a required accuracy or weight distribution.

Artifacts include the actual data, JSON inference state, history, diagnostics, and SVG figures. Full local parameters and histories remain under the ignored `runs/` directory; the repository contains only compact result summaries and figures. The non-neural experiments in 00/07 and the interactive training in 15 use task-specific records rather than forcing every result into a classification-loss format.
