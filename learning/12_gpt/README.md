# 12 · GPT: causal next-token prediction and generation

A minimal teaching experiment is implemented and runnable. Prerequisites: 07, 11. Next chapter: [13_generative](../13_generative/README.md).

Reuse the causal pre-LN block from 11 and token-ID concepts from 07. The default experiment needs no external corpus: a six-symbol cyclic language, 64×8 inputs with shifted next-token targets, and independently sampled train/validation/test starting points. With potentially only six cyclic patterns, this checks architecture rather than natural-language ability or generalization to unseen rules.

Network: vocab 6, context 8, one layer, two heads, width 8, dropout 0.1, and 1048 parameters. Training uses AdamW, gradient clipping, and warmup/cosine scheduling. Test cross-entropy becomes perplexity=`exp(loss)`. Default prediction uses top-1; a temperature 0.7/top-2 example follows training.

```ruby
logits = model.call(input_ids) # [B,T,V]
loss = masked_cross_entropy(logits, next_token_targets)
# Generation: crop to context length, take the last logits, scale by temperature, then sample
```

Generate enters eval/no_grad automatically and restores the original training mode. It validates temperature>0 and legal top_k values, cropping context when the window is too long. Deterministic tests check causal attention for future leakage. Longer generation may still repeat, which is part of the observed results.

The historical custom-corpus entry point is retained in `train_text.rb`:

```bash
bundle exec ruby learning/12_gpt/train_text.rb \
  --data data/learning/song.txt --tokenizer byte --iters 200 --device cpu
```

`train.rb` forwards explicit historical options such as `--data/--tokenizer/--iters/--prompt`. The minimal default experiment uses the generic teaching Loop; the historical Trainer retains Torch AdamW and differs from this round's JSON training-state resume contract. It should not be described as fully resumable training under that contract.

The small GPT teaches target shifting, causality, probabilistic loss, training/generation differences, and state reuse. Factual correctness and semantic ability require separate evaluation.

## Data, training, and independent inference

Run commands from the repository root and install dependencies with `bundle install`. Training and model inference default to `auto`: prefer CUDA and fall back to CPU when unavailable. You can also select `--device cpu` explicitly. These experiments use small synthetic datasets and require no model downloads. Limiting threads reduces CPU overhead for tiny tensors:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/12_gpt/data.rb
bundle exec ruby learning/12_gpt/train.rb --steps 60
bundle exec ruby learning/12_gpt/predict.rb
bundle exec rake test:learning
```

Use `--seed` to change the random seed and `--output` to separate experiment directories. Training also accepts `--device cpu/cuda/auto`. The default output is `runs/learning/12_gpt/default/`; rerunning overwrites artifacts with the same names. `data.rb` exports samples from the data recipe for inspection. The training entry point calls the generators directly and does not depend on that JSON file. Experiment-specific shifts and masks are documented in the experiment source and the actual `data.json`.

For inference, `--model PATH` selects a saved model. `--input PATH` accepts a JSON file containing `{"input": ...}`; the default is a small example with the required shape. Seq2seq/EncoderDecoder perform free-running generation, GPT uses top-1 generation, and other models return scores or reconstructions. RL actor-critic models return policy logits and a value estimate.

## Results and correctness

The [recorded run](results.json) includes the seed, step count, environment, and metrics. The figure below comes from that run; it is not a test acceptance threshold.

![Chapter experiment results](images/gpt-loss.svg)

[Experiment code](../lib/easy_ai_learning/gpt/experiment.rb) connects the steps; shared data generators are in [course/data.rb](../lib/easy_ai_learning/course/data.rb). Core checks are in the [tests](../test/course/gpt_test.rb), with additional gradient comparisons in [derivatives_test.rb](../test/course/derivatives_test.rb). Tests check deterministic formulas, shapes, masks, gradients, state, and parameter updates. They do not train toward a required accuracy or weight distribution.

Artifacts include the actual data, JSON inference state, history, diagnostics, and SVG figures. Full local parameters and histories remain under the ignored `runs/` directory; the repository contains only compact result summaries and figures. The non-neural experiments in 00/07 and the interactive training in 15 use task-specific records rather than forcing every result into a classification-loss format.
