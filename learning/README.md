# AI Learning Curriculum

This curriculum follows mathematics and data → a minimal neural network → training and diagnostics → representation learning and vision → sequences and language → generation, transfer, and decision-making → a capstone. Each chapter explains how the model computes, how its objective is defined, how parameters learn, and how results are checked. Chapter numbers suggest a reading order; choose branches according to their prerequisites.

Chapters 00–16 all have executable data/training experiments and learning documentation. Chapter 01 retains the complete XOR lesson. Later chapters reuse its foundations, followed by training, diagnostics, convolution, recurrent-state, and attention components. Training and neural inference default to `auto`, preferring CUDA and falling back to CPU when unavailable; `--device cpu` selects CPU explicitly. Classical Ruby mathematics and tokenizer-rule learning do not themselves require a GPU.

Each chapter provides a small synthetic task, reproducible data export, independent inference, and actual result JSON and figures. Large-dataset benchmarks, full official BERT/MAE configurations, RoPE/ViT, and similar topics are reading extensions. Default experiments are minimal versions suitable for hand calculations and inspection.

For printing, use the [compact printable course](../docs/learning-course-print.md): core concepts, formulas, and essential code without run logs or large figures.

## Read in your browser

Use the local Jekyll + Just the Docs site for cross-chapter links, search, source browsing, and chapter navigation. See the [installation and reading guide](../docs/learning-site.md). After the first-time setup, run `bin/learning-docs serve` from the repository root and open http://127.0.0.1:8001/.

## Chapters and prerequisites

| Directory | Core topics | Minimal experiment / learning outcome | Prerequisites | Implementation |
| --- | --- | --- | --- | --- |
| [00_foundations](00_foundations/README.md) | Tensors, derivatives, probability, data splits, classical ML baselines | Linear/logistic regression, PCA, and metrics | None | Implemented |
| [01_basic_nn](01_basic_nn/README.md) | Perceptrons, MLP, ReLU, chain rule | Nine-parameter XOR; manual gradients versus autograd | 00 | Runnable experiment and figures |
| [02_training](02_training/README.md) | SGD → Momentum → Adam → AdamW; regularization and training loops | Compare optimizers, learning rates, and dropout on the same MLP | 01 | Optimizers, training controls, and comparisons |
| [03_diagnostics](03_diagnostics/README.md) | Weights, activations, gradients, updates, and generalization diagnostics | Layer-wise statistics; investigate training failures and remedies | 02 | Statistics, histograms, singular values, and failure comparisons |
| [04_autoencoder](04_autoencoder/README.md) | Auto-encoding, bottlenecks, reconstruction, sparsity, denoising | Low-dimensional compression and reconstruction from masked inputs | 01–03 | Implemented |
| [05_cnn](05_cnn/README.md) | Convolution, parameter sharing, receptive fields, pooling, LeNet | Small-image classification; compare MLP/CNN | 01–03 | Implemented |
| [06_resnet](06_resnet/README.md) | Deep optimization, residual connections, BatchNorm | Compare plain and residual CNNs at the same depth | 05 | Implemented |
| [07_tokenizers](07_tokenizers/README.md) | Character/byte/BPE tokenization, Embedding, data leakage | Encoding/decoding round trips; compare vocabularies and sequence lengths | 00–02 | Runnable character/byte/BPE and Embedding experiments |
| [08_rnn](08_rnn/README.md) | RNN → LSTM/GRU, BPTT, gradient clipping | Sequence-pattern prediction and long-range memory | 01–03; add 07 for text tasks | Implemented |
| [09_seq2seq](09_seq2seq/README.md) | Encoder/Decoder, teacher forcing | Sequence reversal; compare training and free-running generation | 08 | Implemented |
| [10_attention](10_attention/README.md) | Additive attention → Q/K/V, multiple heads, masks | Retrieval/alignment, attention matrices, and future-leakage checks | 09 | Additive/multi-head attention, masks, and comparisons |
| [11_transformer](11_transformer/README.md) | Positional encoding, residuals, LayerNorm, FFN | Encoder/Decoder and position/normalization ablations | 10; see 06 for residuals | Encoder, Decoder, ablations, and free-running generation |
| [12_gpt](12_gpt/README.md) | Decoder-only models, next-token prediction, sampling | Autoregressive training and generation on a small corpus | 07, 11 | Small-language experiment; custom-corpus entry point retained |
| [13_generative](13_generative/README.md) | VAE → GAN → Diffusion; optional BERT/MAE masked reconstruction | Compare reconstruction/generation and diagnose generation failures | 04; add 05 for CNN tasks and 11–12 for text | Implemented |
| [14_transfer_learning](14_transfer_learning/README.md) | Pretraining, freezing, full fine-tuning, LoRA, distillation | Compare training from scratch with transfer | 02–03 and the chosen model | Implemented |
| [15_rl](15_rl/README.md) | Bandit → MDP → Q-learning/DQN → policy gradients/PPO | Rewards, returns, exploration, and policy evaluation | 01–03; GPT is not required | Bandit/Q/DQN/REINFORCE/AC/PPO |
| [16_capstone](16_capstone/README.md) | Complete data/model/training/diagnostics/inference workflow | Reproducible experiment reports and model reuse | Chosen branch | Implemented |

Complete 00–03 first, then study a simple fully connected Autoencoder. Follow 05–06 for vision or 07–12 for language. Choose topics within 13 by task. Chapters 14 and 15 are transfer and interactive-learning branches, not necessarily larger networks than GPT.

## Learning model architectures together with training techniques

| Technique | First introduced | Models for review and why |
| --- | --- | --- |
| SGD, Momentum, Adam, AdamW | 02, using the same data/MLP | 05–06 for vision comparisons; 12 for the existing AdamW text trainer |
| Initialization, learning rates, schedulers, warmup | Principles in 02; failure diagnosis in 03 | 06 for deep gradient flow; 11–12 for training stability |
| Dropout, weight decay, early stopping | 02, MLP with independent validation | 04 to avoid mere memorization; 11–12 for attention/FFN dropout |
| Input corruption / random masking | 04, Denoising Autoencoder | 13, BERT/MAE masked prediction; masking forms part of the objective |
| Padding masks / loss masks | 08–09, variable-length batches | 10–12: visibility and scored positions must be handled separately |
| Causal masks | 10 | 12, prevent access to future tokens and align training with generation |
| Data standardization, BatchNorm, LayerNorm | Concepts in 02; observations in 03 | 05–06 for BatchNorm; 11 for LayerNorm; do not conflate them with weight normalization |
| Gradient clipping, accumulation, mixed precision | Concepts in 02; clipping experiments in 08 | 12 for somewhat larger training; establish full-precision correctness first |
| Parameter groups, freezing, low-rank updates | 14 | ResNet/GPT transfer; use 03 to inspect actual updates |

For every mask, specify what it hides, where it applies, and why. Random activation masks in dropout, input corruption, attention-visibility masks, and loss masks have different semantics; they are not interchangeable forms of regularization.

## Existing code and entry points

Run all commands from the repository root. Teaching implementations use the separate `EasyAILearning` namespace; the production library remains in `lib/easy_ai/`.

```text
learning/
├── 00_foundations/ … 16_capstone/  Chapter READMEs and chapter-local scripts
├── 01_basic_nn/{logic,train,predict,plot}.rb
├── 07_tokenizers/{scratch,sentence_piece}.rb  Historical experiments
├── 12_gpt/train.rb
├── lib/easy_ai_learning/           Shared domains without chapter numbers
│   ├── basic_nn/                  Logic, Torch MLP/SGD, manual gradients, reports/plots
│   ├── attention/                 causal_self_attention.rb
│   ├── transformer/               block, feed_forward, positional_embeddings
│   ├── gpt/                       model, config, trainer, batch/text helpers
│   ├── tokenizers/                word/byte BPE, optional Qwen, etc.
│   └── utils/                     Tensor helpers
├── test/                          basic_nn, gpt, tokenizers
└── LEGACY_README.md                Historical record; original commands are not current entry points
```

```bash
bundle exec ruby learning/01_basic_nn/logic.rb
bundle exec ruby learning/01_basic_nn/train.rb
bundle exec ruby learning/01_basic_nn/predict.rb
bundle exec ruby learning/01_basic_nn/plot.rb  # Requires gnuplot
bundle exec ruby learning/12_gpt/train_text.rb \
  --data data/learning/song.txt --tokenizer byte --iters 200 --device auto
bundle exec rake test:learning
```

XOR prefers CUDA by default and falls back to CPU when unavailable; `--device cpu` forces CPU. No model download or corpus preparation is needed. The recorded default-seed-1337 CUDA run took 436 updates, reached a maximum absolute error of 0.009843, and predicted all 4/4 truth-table entries correctly. This fits the complete training table rather than establishing generalization. See [01](01_basic_nn/README.md) for details, parameter reuse, and existing figures.

XOR writes inference parameters, loss, and figures to the ignored `runs/learning/basic_nn/logic-gates/`. These are not training-resume checkpoints with optimizer state. Rerunning overwrites the default output; use `--output` for comparisons. GPT's local corpus resides in the ignored `data/learning/`, selected through `--data` or `EASY_AI_DATA`; prepare it separately in a new environment. `bin/train_basic.rb` points to `12_gpt/train.rb`.

Directory migration: `02_rnn → 08_rnn`, `03_seq2seq → 09_seq2seq`, `04_attention → 10_attention`, `05_transformer → 11_transformer`, `06_gpt → 12_gpt`, `07_rl → 15_rl`, `tokenizers → 07_tokenizers`. Shared-library names, test locations, and existing artifact paths do not move with chapter numbering.

## Completion standard for each chapter

Start with a small example that can be calculated by hand, then provide a Ruby/Torch.rb experiment covering prerequisites and goals, forward computation and tensor shapes, loss/backward, parameter counts, training/inference differences, minimal data, executable commands, learning curves, and failure cases. Distinguish minimal executable tasks from architectural extensions offered only as reading.

Record data splits, seeds, configuration, update counts, and evaluation metrics. Control variables in comparisons, use multiple seeds, and report variation. First check whether a small batch can be fitted, then evaluate independent samples and perform ablations. Finite truth tables, training-set reconstruction, and training loss cannot replace generalization evaluation.

XOR terminal plots use `unicode_plot`, and publication figures use gnuplot. New chapters generate SVG directly without extra plotting dependencies. Shared implementations live in domain modules such as `foundations/`, `training/`, `diagnostics/`, `autoencoder/`, `cnn/`, `resnet/`, `rnn/`, `seq2seq/`, `generative/`, `transfer/`, and `rl/`; chapter entry points connect the experiments. Training artifacts live under the ignored `runs/learning/<topic>/<experiment>/`.

## An executable learning path

```bash
# Install the existing Gemfile dependencies; no extra corpus or pretrained-model downloads
bundle install
bundle exec ruby learning/run_all.rb
# Explicit CPU comparison; unit tests also use CPU and do not depend on GPU hardware
bundle exec ruby learning/run_all.rb --device cpu --output runs/learning/cpu-course
bundle exec rake test:learning
```

`run_all.rb` exports data and trains in chapter order, 00–16. Defaults are 60 full-batch updates per comparison model, 60 episodes for RL, and XOR's original 10,000-update limit/error target. Later chapters repeatedly reuse earlier components. Each chapter defaults to `runs/learning/<chapter>/default/`, with an aggregate summary at `runs/learning/run-summary.json`. A full rerun is a teaching experiment, not a unit test; actual records are retained even when training outcomes are uncertain.

Each chapter independently runs `data.rb`, `train.rb`, and `predict.rb`. New chapter inference accepts `--model PATH` and `--input JSON_FILE`, with valid small default inputs. Chapter 00 uses `--value`, 07 uses `--text`, and 01 retains its original options. Neural inference also supports `--device`. Setting `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` reduces threading overhead for small CPU tensors; the all-course runner sets them automatically.

Core tests check hand-calculated/closed-form formulas, finite differences/autograd, convolutions, gates, BPTT detach, masks/future leakage, freezing/LoRA, GAN gradient isolation, DDPM/GAE/PPO objectives, exact parameter updates, and state round trips. Passing does not require training accuracy, a convergence step count, decreasing loss, or a particular-looking weight distribution.

The generic teaching optimizer/Loop supports JSON state independently. The Loop's per-update seeds, identical full-batch data/objective, and fixed total_steps enable exact resume. This contract does not cover the historical Torch Trainer or arbitrary-point restoration of RL environments/rollouts/replay. EarlyStopping saves the best model for inference; the final optimizer state is not the state from that best step.

Raw model/optimizer parameters and full datasets remain in ignored runs. Each chapter's `results.json` and `images/` are compact snapshots from actual runs, including device, seed, and step count. Interpret results within the sample/task scope; perfect scores on simple synthetic data do not establish real-world AI capabilities.

## CUDA environment record

All 00–16 experiments completed using the default `auto` setting. Experiments with neural tensors used CUDA; pure Ruby mathematics and tokenizer rules still ran in Ruby. The tool sandbox cannot access the driver, so CPU unit tests and CUDA experiments outside the sandbox were validated separately. Unavailable libraries/initialization failures follow the existing DevicePolicy CPU fallback. Arbitrary runtime errors were not swallowed and reported as successful CUDA runs.

This machine's default system cuDNN reported `Invalid handle / cublasLtGetVersion` on the CNN path. Selecting an already installed compatible cuDNN 9 library allowed CNN forward/backward and all CUDA experiments to pass without changing the system installation. To use the same setup:

```bash
bundle exec ruby learning/run_all.rb --device auto \
  --cudnn-dir /home/andersen/Installed/libtorch-2.10.0-cu128/lib \
  --output runs/learning/cuda-course
```

`--cudnn-dir` links only that directory's `libcudnn*.so.9` files into the current output's `runtime/cudnn/`. It prepends the compatible-library path for child processes while preserving the original LD_LIBRARY_PATH, avoiding replacement of other LibTorch libraries. This is an optional machine-specific setting, not a universal device requirement; compatible environments do not need it. Independent inference needs the same compatible-library environment. `--chapter 05_cnn` runs only one chapter.

The optimizer's `kind` in the [explicit numerical/state implementation](lib/easy_ai_learning/training/optimizer.rb), RNN `kind`, MLP `activation`, classical-ensemble `kind`, and device `requested` all accept string/symbol values. Unknown enum values raise ArgumentError. [Option contract tests](test/course/options_test.rb) verify equivalent behavior for both forms.

Inspect coverage with `bundle exec ruby learning/verify.rb --coverage`; the report is in the ignored `tmp/learning/coverage.json`. Coverage helps identify untested branches; core logic correctness still rests on hand calculations, finite differences, and behavior contracts.
