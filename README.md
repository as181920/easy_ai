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
# Example: train on every *.txt under data/song_corpus with byte BPE
bundle exec ruby bin/train_basic.rb \
  -d data/song_corpus \
  --tokenizer byte \
  -m 2000 \
  -B 64 \
  -i 400

# Example: train on a single file with a word-level tokenizer
bundle exec ruby bin/train_basic.rb \
  -d data/xiaojing.txt \
  --tokenizer word \
  -b 32 \
  -B 2 \
  -m 40 \
  -i 5
```

Key behaviors:

- **Device selection** – defaults to CUDA when available (tested on 6 GB RTX 3000); otherwise uses CPU. If CUDA throws an error mid-run, the script automatically retries on CPU.
- **Tokenizer training** – if the tokenizer only contains the `<|unk|>` entry, the script auto-trains it on the provided corpus before building the dataset.
- **Flexible `-d` input** – pass a single file or a directory; directories are scanned recursively for `*.txt` files and concatenated.
- **Configuration knobs** – every CLI flag mirrors an environment variable (`EASY_AI_*`). Run `bundle exec ruby bin/train_basic.rb --help` for the full list.
- **Output** – after training it prints a Unicode loss chart and a greedy sample seeded by `--prompt` (default: `人间有味是清欢`).

### CLI Parameters

| Flag | Environment | Description | Default |
| --- | --- | --- | --- |
| `-d, --data PATH` | `EASY_AI_DATA` | Path to a single text file or a directory (recursively scans for `*.txt`). | `data/song.txt` |
| `-t, --tokenizer TYPE` | `EASY_AI_TOKENIZER` | `word` or `byte`, selecting `WordBpe` or `ByteBpe`. | `word` |
| `-m, --merges N` | `EASY_AI_MERGES` | Number of BPE merge operations during tokenizer training. | `2000` |
| `-f, --min-freq N` | `EASY_AI_MIN_FREQ` | Minimum pair frequency before a merge is accepted. | `2` |
| `-b, --block-size N` | `EASY_AI_BLOCK_SIZE` | Sequence length (context window) for GPT. | `64` |
| `-l, --layers N` | `EASY_AI_LAYERS` | Number of transformer blocks. | `2` |
| `-H, --heads N` | `EASY_AI_HEADS` | Attention heads per block. | `2` |
| `-e, --embed N` | `EASY_AI_EMBED` | Embedding/hidden dimension. | `128` |
| `-B, --batch N` | `EASY_AI_BATCH` | Batch size (in samples) per iteration. | `16` |
| `-i, --iters N` | `EASY_AI_ITERS` | Training iterations (steps). | `200` |
| `--device DEVICE` | `EASY_AI_DEVICE` | `cuda` or `cpu`. CUDA auto-fallback is enforced. | auto-detect |
| `-p, --prompt TEXT` | `EASY_AI_PROMPT` | Prompt used for post-training sampling. | `人间有味是清欢` |

Additional options (`EASY_AI_LR`, `EASY_AI_WEIGHT_DECAY`, `EASY_AI_LOG`, `EASY_AI_GRAD_CLIP`, `EASY_AI_SEED`, etc.) mirror the config defaults in `EasyAI::Config`. Use the env vars to tweak optimizer settings when running long experiments.

## Tests

Tokenizers mirror the `lib/` layout under `test/easy_ai/`:

```bash
bundle exec ruby -Itest test/easy_ai/tokenizers/word_bpe_test.rb
bundle exec ruby -Itest test/easy_ai/tokenizers/byte_bpe_test.rb
```

Add future tests under `test/easy_ai/{data,modules,models,...}` to keep parity with the library tree.

## Acknowledgements

- **torch-rb** by [Andrew Kane](https://github.com/ankane) and contributors brings PyTorch ergonomics to Ruby—this project would not exist without it.
- **unicode_plot** by the [Red Data Tools](https://github.com/red-data-tools/unicode_plot.rb) makes it easy to visualize training curves directly in the terminal.

## Next Steps

1. Add CLI commands for dataset prep / sampling so the training script can optionally skip optimizer steps.
2. Write smoke tests for `EasyAI::Data::Batcher` and the GPT forward pass shapes.
3. Persist trained tokenizers/models from `bin/train_basic.rb` so experiments can resume without retraining.
4. Implement a preprocessing pipeline (scheme #1) that tokenizes large corpora offline into chunked binary files so training can memory-map or stream IDs instead of loading entire directories.
