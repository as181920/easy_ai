# Notes

## tokenization prepare

Pre-tokenization: 强制在中文和英文之间、甚至每个中文字符之间加上分割符，防止 BPE 把跨度太大的内容强行合并。
字节回退： 确保词表包含 256 个基础字节。这样即使是 Unicode 动态更新了新表情包或极罕见古籍字，模型也能用 3-4 个字节 Token 强行表示出来。

## num_merges

bpe进行pair的次数，建议合并次数设定在 8,000 - 16,000 左右，这通常能覆盖绝大多数常用的中文词汇和短语。
