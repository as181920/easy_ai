# 07 · Tokenizer、BPE 与 Embedding

已实现可运行的最小教学实验。先修：00–02。 下一章：[08_rnn](../08_rnn/README.md)。

顺序：字符 → UTF-8 字节 → 高频相邻对合并/BPE → token ID → Embedding。Tokenizer 学的是规则/词表；Embedding 通过后续任务的梯度学向量，二者不是同一类“训练”。

本章新增 Character 与 ReversibleBpe：Character 只从训练文本建表，未知字符输出 UNK；字节版本固定 256 基础符号；BPE 在整数 byte ID 上学习 24 次合并，保留空白、换行和未见 UTF-8 字节。保存格式包含全部 merges，加载后逐次应用。

```ruby
pairs = tokens.each_cons(2).tally
best = pairs.max_by { |pair, frequency| frequency }.first
# 从左到右进行不重叠替换，新 ID=256+merge_index
```

规则只由训练语料决定，验证文本包含新符号与 emoji，展示 Character 的 OOV 和 Byte/BPE 的可逆性；比较词表规模、token 长度和 8 维 Embedding 参数量。短 token 序列不自动代表更好的语言模型。Embedding 另有 `[1,2,1]` 查表梯度演示：row 1 梯度加两次、row 2 加一次，未查行不更新。

历史 `scratch.rb`、`sentence_piece.rb` 和 word-splitting ByteBpe/WordBpe 保留供对照；历史实现有空白归一化、Unicode/OOV 或算法近似限制，不作为本章可逆字节 BPE 的实现。SentencePiece 近似也不是官方算法完整复现。

共享可逆 BPE 独立于生产 tokenizer；12 的最小 GPT 使用显式六符号词表，历史自定义语料入口仍可选原 Byte/Word/Qwen tokenizer。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/07_tokenizers/data.rb
bundle exec ruby learning/07_tokenizers/train.rb --steps 60
bundle exec ruby learning/07_tokenizers/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/07_tokenizers/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--text "新文本"` 编码/解码。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/token-length.svg)

[实验代码](../lib/easy_ai_learning/tokenizers/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/sequence_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
