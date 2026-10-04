# 13 · VAE, GAN, Diffusion, and masked reconstruction

A minimal teaching experiment is implemented and runnable. Prerequisites: 04; add 05 for images and 11 for text/patch models. Next chapter: [14_transfer_learning](../14_transfer_learning/README.md).

Reuse AE encoders/decoders, MLP projections, Transformer representations, and training foundations from 02. Each experiment has its own loss, so raw loss values cannot rank different model families.

| Model | Implemented experiment | Objective and diagnostics |
| --- | --- | --- |
| VAE | 4D manifold→μ/logvar→2D latent→reconstruction | Reconstruction summed over dimensions + β KL; evaluate reconstruction and prior sampling separately |
| GAN | Two groups of 2D points, MLP generator/discriminator | D uses detached fake samples; G uses non-saturating BCE; inspect mode collapse |
| DDPM | 12-step linear beta schedule, 2D noise predictor, stepwise reverse sampling | Noise MSE; posterior variance is zero at the last reverse step |
| Masked language | Bidirectional Encoder, token 7 as MASK, loss only at masked positions | Random training masks and fixed validation masks; not a full reproduction of BERT pretraining |
| MAE | Split 8×8 images into sixteen 2×2 patches; the encoder reads only eight visible patches | Restore positions with mask tokens; the decoder reconstructs with loss only on hidden patches |

```text
VAE: z=μ+exp(logvar/2)*ε
KL = -0.5 mean_batch sum_dim(1+logvar-μ²-exp(logvar))
DDPM: x_t=sqrt(ᾱ_t)x_0+sqrt(1-ᾱ_t)ε
ε_θ=MLP(concat(x_t,t/(T-1)))
```

To keep calculations simple, MAE shares sampled visible indices across a batch. Hidden image pixels are never passed to the encoder. Patchify/unpatchify round-trip exactly. Tests change hidden patches to verify that predictions do not depend on their contents.

The default short runs on small datasets may show limited coverage, collapse, or poor reconstruction. Data/generated-sample figures and raw sample JSON are retained. VAE β=0.1 is a teaching variant with a weighted KL, not the original β=1 ELBO. A fixed-scale Gaussian corresponds to the squared-error term; output variance is not estimated here. Decreasing GAN losses or diffusion noise loss alone does not establish sample quality.

Sources: [VAE](https://arxiv.org/abs/1312.6114), [DDPM](https://arxiv.org/abs/2006.11239).

## Data, training, and independent inference

Run commands from the repository root and install dependencies with `bundle install`. Training and model inference default to `auto`: prefer CUDA and fall back to CPU when unavailable. You can also select `--device cpu` explicitly. These experiments use small synthetic datasets and require no model downloads. Limiting threads reduces CPU overhead for tiny tensors:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/13_generative/data.rb
bundle exec ruby learning/13_generative/train.rb --steps 60
bundle exec ruby learning/13_generative/predict.rb
bundle exec rake test:learning
```

Use `--seed` to change the random seed and `--output` to separate experiment directories. Training also accepts `--device cpu/cuda/auto`. The default output is `runs/learning/13_generative/default/`; rerunning overwrites artifacts with the same names. `data.rb` exports samples from the data recipe for inspection. The training entry point calls the generators directly and does not depend on that JSON file. Experiment-specific shifts and masks are documented in the experiment source and the actual `data.json`.

For inference, `--model PATH` selects a saved model. `--input PATH` accepts a JSON file containing `{"input": ...}`; the default is a small example with the required shape. Seq2seq/EncoderDecoder perform free-running generation, GPT uses top-1 generation, and other models return scores or reconstructions. RL actor-critic models return policy logits and a value estimate.

## Results and correctness

The [recorded run](results.json) includes the seed, step count, environment, and metrics. The figure below comes from that run; it is not a test acceptance threshold.

![Chapter experiment results](images/gan-distribution.svg)

[Experiment code](../lib/easy_ai_learning/generative/experiment.rb) connects the steps; shared data generators are in [course/data.rb](../lib/easy_ai_learning/course/data.rb). Core checks are in the [tests](../test/course/generative_test.rb), with additional gradient comparisons in [derivatives_test.rb](../test/course/derivatives_test.rb). Tests check deterministic formulas, shapes, masks, gradients, state, and parameter updates. They do not train toward a required accuracy or weight distribution.

Artifacts include the actual data, JSON inference state, history, diagnostics, and SVG figures. Full local parameters and histories remain under the ignored `runs/` directory; the repository contains only compact result summaries and figures. The non-neural experiments in 00/07 and the interactive training in 15 use task-specific records rather than forcing every result into a classification-loss format.
