# easy_ai

Ruby playground for learning large-language-model internals. The repo currently focuses on tokenizer experimentation (word-level BPE + byte BPE) and a Torch‑RB GPT mini-stack that can be trained on classical Chinese corpora.

## Project Roadmap

1. **Core namespace** – `lib/easy_ai.rb` bootstraps Zeitwerk, config management, and shared utils so every component can rely on the same Torch + ActiveSupport setup.
2. **Tokenizers** – `EasyAI::Tokenizers::{WordBpe,ByteBpe}` expose a consistent `train/encode/decode/save/load` interface, keep `<|unk|>` reserved, and can serialize merges for reuse.
3. **Data pipeline** – `EasyAI::Data::TextDataset` and `Batcher` convert raw text into contiguous token IDs and produce `[batch, seq_len]` tensors with deterministic seeding.
4. **Model modules** – attention, feed-forward, transformer blocks, positional embeddings, and layer norms built with Torch‑RB.
5. **Model assembly** – `EasyAI::Models::GPT` wires embeddings + N transformer blocks and provides `generate` for autoregressive sampling.
6. **Training utilities** – `EasyAI::Trainers::Trainer` uses AdamW, grad clipping, and loss logging callbacks.
7. **CLI scripts** – `bin/train_basic.rb` offers an end-to-end demo: train a tokenizer (if needed), load data, train a GPT, plot the loss curve via `unicode_plot`, and sample text.
8. **Tests & docs** – tests mirror the library structure under `test/easy_ai/**`; this README now holds the living documentation that used to be in `notes.md`.

## Installation

```bash
bundle install
```

The project depends on:

- Ruby ≥ 3.0
- [torch-rb](https://github.com/ankane/torch-rb) (CPU build works; CUDA build unlocks GPU training)
- `unicode_plot` for inline loss visualizations

## Training Demo (`bin/train_basic.rb`)

This script stitches tokenizer → dataset → GPT → trainer and displays a Unicode loss plot plus a sampled continuation.

```bash
# Example: train on Song lyrics with byte BPE and a tiny GPT
bundle exec ruby bin/train_basic.rb \
  -d data/song.txt \
  --tokenizer byte \
  -m 2000 \
  -B 64 \
  -i 400
```

Key behaviors:

- **Device selection** – defaults to CUDA when available (tested on 6 GB RTX 3000); otherwise uses CPU. If CUDA throws an error mid-run, the script automatically retries on CPU.
- **Tokenizer training** – if the tokenizer only contains the `<|unk|>` entry, the script auto-trains it on the provided corpus before building the dataset.
- **Configuration knobs** – every CLI flag mirrors an environment variable (`EASY_AI_*`). Run `bundle exec ruby bin/train_basic.rb --help` for the full list.
- **Output** – after training it prints a Unicode loss chart and a greedy sample seeded by `--prompt` (default: `人间有味是清欢`).

## Tests

Tokenizers mirror the `lib/` layout under `test/easy_ai/`:

```bash
bundle exec ruby -Itest test/easy_ai/tokenizers/word_bpe_test.rb
bundle exec ruby -Itest test/easy_ai/tokenizers/byte_bpe_test.rb
```

Add future tests under `test/easy_ai/{data,modules,models,...}` to keep parity with the library tree.

## Acknowledgements

- **torch-rb** by [Andrew Kane](https://github.com/ankane) and contributors brings PyTorch ergonomics to Ruby—this project would not exist without it.
- **unicode_plot** by [Red Data Tools](https://github.com/red-data-tools/unicode_plot.rb) and collaborators makes it easy to visualize training curves directly in the terminal.

## Next Steps

1. Add CLI commands for dataset prep / sampling so the training script can optionally skip optimizer steps.
2. Write smoke tests for `EasyAI::Data::Batcher` and the GPT forward pass shapes.
3. Persist trained tokenizers/models from `bin/train_basic.rb` so experiments can resume without retraining.
