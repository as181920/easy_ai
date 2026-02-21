# Notes

## tokenization prepare

Pre-tokenization: 强制在中文和英文之间、甚至每个中文字符之间加上分割符，防止 BPE 把跨度太大的内容强行合并。
字节回退： 确保词表包含 256 个基础字节。这样即使是 Unicode 动态更新了新表情包或极罕见古籍字，模型也能用 3-4 个字节 Token 强行表示出来。

## num_merges

bpe进行pair的次数，建议合并次数设定在 8,000 - 16,000 左右，这通常能覆盖绝大多数常用的中文词汇和短语。

# Project Plan

## 1. Core Namespace

- 添加 `lib/easy_ai.rb` 和 `lib/easy_ai/version.rb`，让 Bundler 与测试统一加载命名空间。
- `test_helper.rb` 等基础文件通过该入口引用库代码。

## 2. Tokenizer Layer

- 将现有 BPE 实现在 `lib/easy_ai/tokenizers/` 内部整理，提供 `WordBpe`、`ByteBpe` 等适配器。
- 抽象 `BaseTokenizer` 接口，统一 `train/encode/decode/save/load` 流程。
- 编写词表序列化工具与 `bin/train_tokenizer.rb` 脚本，支持从语料快速训练/导出。

## 3. Data Pipeline

- `lib/easy_ai/data/text_dataset.rb`：负责读取原始文本、调用分词器并生成连续 token ID。
- `lib/easy_ai/data/batcher.rb`：按 `[seq_len, batch_size]` 产出 Torch tensor，包含 shuffle、train/val split。
- 公共工具（chunk、seed）整理在 `lib/easy_ai/data/data_utils.rb`。

## 4. Modules & Model

- `lib/easy_ai/modules/`：实现 `attention.rb`、`feed_forward.rb`、`transformer_block.rb`、`positional_embeddings.rb` 等组件。
- `lib/easy_ai/models/gpt.rb`：拼装嵌入、N 个 transformer block、语言建模头，并提供 `generate`。
- `lib/easy_ai/config.rb`：集中管理超参，支持 YAML 覆盖。

## 5. Training Utilities

- `lib/easy_ai/trainers/trainer.rb`：封装 AdamW、梯度裁剪、checkpoint、cosine lr 等训练细节。
- `lib/easy_ai/trainers/callbacks/logger.rb` 等回调，用于输出训练指标。

## 6. CLI Scripts

- 新增 `bin/prepare_dataset.rb`、`bin/train.rb`、`bin/sample.rb` 等脚本驱动完整流程。
- 所有脚本加载配置，实例化 tokenizer/dataset/model/trainer，保证命令行体验。

## 7. Tests & Docs

- 测试按照命名空间重组（如 `test/tokenizers/*`、`test/data/*`）。
- README/notes 将覆盖完整 workflow：训练分词器 → 预处理数据 → 训练模型 → 文本采样。
