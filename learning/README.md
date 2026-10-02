# Learning progression

Teaching code uses the independent `EasyAILearning` namespace. It is not automatically loaded by the production Decision model. Lessons progress from scalar calculations to sequence models, attention and GPT; a planned directory does not mean that lesson is implemented.

```text
learning/
|-- 01_basic_nn/       implemented: branches, perceptrons, ReLU MLP, training, plots
|   |-- logic.rb       fixed logic and mathematical existence proof
|   |-- train.rb       learn XOR from random coefficients on Torch/CUDA
|   |-- predict.rb     load saved parameters without training
|   |-- plot.rb        export grouped figures and both raw-score views
|   `-- README.md
|-- 02_rnn/            planned: recurrent state and backpropagation through time
|-- 03_seq2seq/        planned: encoder/decoder and teacher forcing
|-- 04_attention/     causal attention component exists; standalone lesson planned
|-- 05_transformer/   block components exist; standalone lesson planned
|-- 06_gpt/           existing autoregressive text-training example
|   `-- train.rb
|-- tokenizers/       standalone historical tokenizer experiments
|-- lib/easy_ai_learning/
|   |-- basic_nn/     explicit logic, Torch MLP/SGD, scalar gradient reference, reporting
|   |-- attention/    causal_self_attention.rb
|   |-- transformer/  block.rb, feed_forward.rb, positional_embeddings.rb
|   |-- gpt/          model.rb, config.rb, trainer.rb, text/batch helpers
|   |-- tokenizers/   shared educational tokenizer implementations
|   `-- utils/        shared tensor helpers
|-- test/             basic_nn/, gpt/, tokenizers/
`-- LEGACY_README.md   historical notes, preserved unchanged
```

## Start with the basic lesson

```bash
# No model download or corpus; training prefers CUDA and falls back to CPU.
bundle exec ruby learning/01_basic_nn/logic.rb
bundle exec ruby learning/01_basic_nn/train.rb
```

The first command compares explicit `if/else` gates with three fixed perceptrons and an exact ReLU construction. The second trains a dense `2 -> 2 -> 1` XOR network with 9 parameters. Ruby defines the architecture and loop; Torch.rb performs tensor forward calculation, MSE, autograd and SGD on CUDA by default (`--device cpu` forces CPU). The scalar reference retains hand-written forward/backpropagation for checking every derivative. It prints progress, learned equations, the entire truth table, loss plots, ReLU and learned XOR-function slices using `unicode_plot`.

Default seed 1337: 436 CUDA updates, maximum absolute error 0.009843, all four Boolean XOR outputs correct after thresholding at 0.5. This is fitting the complete finite truth table, not a held-out generalization result. All parameters are learned; the exact solution is only a separate demonstration. See [the lesson](01_basic_nn/README.md) for derivation and parameter reuse.

For two README image groups (AND/OR/NAND/XOR references; ReLU/training loss/score heatmap/score surface/prediction), run `bundle exec ruby learning/01_basic_nn/plot.rb` after training (requires gnuplot). See [the published figures](01_basic_nn/README.md#observed-runs-and-plots). Terminal charts remain `unicode_plot`.

`LogicNetwork.load(path, device: :auto)` restores saved inference parameters. `model.score([1, 0])` returns the sole continuous XOR score; `model.predict([1, 0])` returns integer `1`. `bundle exec ruby learning/01_basic_nn/predict.rb` loads and evaluates them without training.

Outputs go to ignored `runs/learning/basic_nn/logic-gates/{model.json,loss.json,plots.txt}`. Re-running overwrites these educational outputs; use `--output` for separate experiments. Other seeds or learning rates may fail; the script reports failure and exits unsuccessfully if its error target is not reached.

## Existing GPT example

```bash
bundle exec ruby learning/06_gpt/train.rb \
  --data data/learning/song.txt --tokenizer byte --iters 200 --device cpu
bundle exec rake test:learning
```

GPT is a decoder-only causal Transformer, so its executable belongs under `06_gpt`, not the generic Transformer stage. The reusable components now live under `EasyAILearning::Attention`, `EasyAILearning::Transformer` and `EasyAILearning::GPT`. The namespace change is intentional; old `Models::GPT`/`Modules` names are removed. `bin/train_basic.rb` forwards to the new GPT entry point.

Local text remains under ignored `data/learning/`. GPT accepts `--data` or `EASY_AI_DATA`; fresh environments must prepare their own corpus. The historical optional Qwen tokenizer is preserved, but the basic lesson and production Decision training do not require it. Original gradient-descent illustrations remain in the repository README.

## Next lessons

Each new stage should provide a derivation, readable forward/backward calculations, a small executable training task, parameter accounting, progress and an interpretable plot. Reuse earlier components where appropriate; distinguish fixed mathematical constructions from trained parameters, training fit from validation, and scores from probabilities. Use `unicode_plot` first; add gnuplot only when an actual visualization requires it.

Production algorithms remain separately readable in `lib/easy_ai/{nn,decision,optim,tokenizers}`. The learning directory can evolve without altering their API.
