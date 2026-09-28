# 本机验收记录

2026-09-28，本地 Ruby 3.4.5、Torch.rb 0.23.0；Quadro RTX 3000，6144 MiB，compute capability 7.5，驱动 595.91.07。以下是流程与容量验收，不是通用模型质量认证。CUDA 检查在可访问 GPU 的主机环境运行。

## 已验证的行为

当前自动检查：正式库 37 tests / 275 assertions；教学代码 7 tests / 14 assertions，均无失败。RuboCop 检查通过。初版的两个命名空间 eager load 已验证。

- CPU 测试覆盖可学习的两样本任务、AdamW 与 LibTorch 对照、带 dropout 的断点续训、数据变更拒绝续训、checkpoint 损坏识别。
- 候选重排、分块与 padding 隔离；混合候选数量的有限梯度；重复 state 缓存；中英日阿等 UTF-8 分词往返。
- 扩层和 FFN 扩宽时原函数保持、新增部分可学习、优化器矩迁移、自动增长拒绝与回滚、消耗步数保留。
- 独立分组校准/评估、CLI 全流程，原学习目录测试单独运行。
- GPU 与 CPU 的 tiny 模型 logits 最大绝对差 `7.45e-8`；GPU 训练 2 步后在 CPU 恢复到第 3 步；将软预算设为 1 MiB 后正确使用 CPU。
- 真实监督 checkpoint 手动扩至 3 层后在 CUDA 恢复到第 12 步，参数从 156,801 变为 165,345。

复现 CPU/CUDA 行为：

```bash
bundle exec rake test
bundle exec rake test:learning
bundle exec rake lint
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bundle exec ruby benchmarks/decision/verify_gpu.rb
```

## 正式配置的实际训练

新增一键流程实跑：`bundle exec ruby bin/easy-ai pipeline --output runs/decision/pipeline-showcase`。
默认 1148 万参数、Ruby BPE（实际词表 3027）、五语言每 split 1000 行 / 200 个源分组、8 候选；MLM 100 步、choice 300 步，全程 CUDA，约 297 秒完成数据准备、训练、校准和 test，随后生成报告。
训练前检查未发现超长截断。图像快照在根 README，完整本地产物位于 `runs/decision/pipeline-showcase/report/`。

```text
MLM validation NLL:       10.08695 (step 10) -> 6.70568 (step 100)
Choice train batch loss:   2.07220 (step 10) -> 1.65842 (step 300)
Choice validation NLL:     2.07787 (step 10) -> 2.07743 (step 300)
Calibration temperature:   1.88620
Calibration NLL:           1.98863 -> 1.91123
Calibration Brier:         0.81929 -> 0.82398
Calibration ECE:           0.03527 -> 0.06977
Test accuracy / NLL:       0.255 / 1.95793
Test Brier / ECE:          0.83761 / 0.05256
```

候选训练后期的验证曲线有明显波动及回升；这里报告最后 checkpoint，没有按 test 挑选权重。Brier/ECE 没有随温度拟合一起改善。
这些是少量平行文本和采样候选上的实验结果。下面保留此前短程容量基准，二者的数据与更新次数不同，不应直接比较效果。

`config/decision/small.yml`：11,484,929 参数，FP32，状态上限 256、候选序列上限 64，choice microbatch 4，梯度累积 8。真实 MASSIVE 五语言短文本、每例四个候选，训练样本 60 条。

```text
任务                GPU 更新    结果
MLM                 4           loss 10.3913，保存可恢复 checkpoint
candidate CE        8           全部在 CUDA 完成

choice 更新时长:     首步 0.612s；后续 0.293--0.355s
含最终 checkpoint:  4.211s
更新后采样显存最大:  498 MiB（当前训练进程）
```

这是短文本动态 padding 下的实际测量，不是全部输入都填满 256/64 tokens 时的峰值。采样包含进程 CUDA 开销但不保证捕获瞬时峰值；候选数、长度、缓存和桌面占用改变后需重新测量，4 GiB 软预算与 CPU 回退仍必要。

复现基准（输出目录应换成新路径）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bundle exec ruby benchmarks/decision/training.rb \
  config/decision/small.yml data/decision/massive-smoke/tokenizer-ruby.json \
  data/decision/massive-validation/train.jsonl runs/decision/recheck-choice choice 8
```

这里 `massive-smoke` 使用 `--limit 40 --candidates 4`，`massive-validation` 使用 `--limit 12 --candidates 4`，均为默认五语言；tokenizer 只在前者的训练 split 训练。命令不会自动生成这些数据，准备步骤见使用指南。计时排除初始模型放置与初始 checkpoint，包含实际 tokenization、GC、AdamW 和后续保存；详情保存在 run 的 `benchmark.json`。

## 小规模端到端质量观测

独立流程采用 smoke 配置：MLM 10 步、监督训练 10 步，再用独立 calibration 60 条拟合温度。数据按五语言共享原始 ID 分组，每 split 只有 12 个不同源样本，因此 60 条不是 60 个独立语义样本。

```text
calibration: T = 0.150812
                    拟合前         拟合后
NLL                 1.370941       1.335788
Brier               0.742514       0.726838
ECE                 0.008089       0.059569
accuracy            0.25           0.25

held-out test (60 rows, 4 sampled choices):
accuracy            0.666667
NLL                 1.269926
Brier               0.695793
ECE                 0.378776
```

温度拟合降低 calibration NLL，但 **ECE 反而变差**。test 样本很少、语言间平行重复、候选为采样负例；这些数字不能用于宣称模型达到某个通用水平或优于随机模型的稳定结论。每种语言分数接近也不能证明学会了五种语言；很短的训练可能主要利用候选先验。后续质量研究应扩充数据，加入只读候选的基线、完整标签候选集、按意图分层的测试和独立业务测试集。

本地验收产物：`runs/decision/acceptance-mlm`、`acceptance-choice`、`acceptance-calibrated`、`acceptance-small-mlm`、`acceptance-small-choice`、`acceptance-grown`。这些目录及原始下载、数据、tokenizer 全部被 gitignore；clone 仓库不会附带权重。

## Tokenizer 取舍的测量

200 条五语言训练例展开为 1200 段文本，同一机器请求词表上限 4096：

| backend | 实际词表 | 训练秒 | 编码全部文本秒 | 编码 tokens | 原文往返 |
| --- | ---: | ---: | ---: | ---: | --- |
| Ruby byte BPE | 1288 | 0.7564 | 0.2006 | 6449 | 通过 |
| tokenizers gem | 1374 | 0.0317 | 0.0149 | 4836 | 通过 |

这组小样本测量中原生 backend 明显更快，适合扩大语料时优先试用。两者预分词不同，输出 token 数不同，不能理解为同一词表上的纯实现语言性能比；embedding 与 tokenizer 必须绑定。Ruby 实现保留为默认学习路径，不承诺大型语料性能。

原始 MASSIVE 1.1 下载约 40.3 MB，SHA256：

```text
4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577
```

来源、许可证、切分和采样策略保存在数据 manifest；[官方 MASSIVE 仓库](https://github.com/alexa/massive) 提供数据说明。

## 改进训练与 GPU 吞吐（2026-09-28）

`config/decision/massive.yml`：11,747,841 参数；本机监督训练 1000 步约 125 秒，全流程约 208 秒（复用已准备数据与 tokenizer）。
最佳 validation 权重第 700 步；test accuracy 75.6%，中文 test 75.0%；详细状态消融、任务限制和仍失败的否定句见 [diagnostics.md](diagnostics.md#改进配置的结果与边界)。

同一 matching 模型、同一数据、自写 AdamW、有效 batch=32，短程各跑 12 次更新，排除前两次的更新耗时：

| microbatch × accumulation | 秒/更新 | 行/秒 | 更新后采样进程显存最大值 |
| --- | ---: | ---: | ---: |
| 4 × 8 | 0.38177 | 83.82 | 514 MiB |
| 16 × 2 | 0.14816 | 215.98 | 700 MiB |
| 32 × 1 | 0.11395 | 280.82 | 1060 MiB |

这测的是短文本实际训练吞吐，不是 GPU 理论算力或显存峰值。更大的 microbatch 减少每步 Ruby 组装、张量调用和强制 GC 次数，实测约快 3.35 倍。
12 步中记录的 GC 时间分别为 1.586 / 0.551 / 0.366 秒。显存检查会启动 nvidia-smi，loss.item 与梯度范数还会同步；因此即使加速，仍可能看到 GPU 利用率波动，不能保证持续 100%。
完整 1000 步期间的一次采样：训练进程约 1850 MiB、整卡约 2627 MiB、GPU-util 35%、功耗 77/80 W；单次 GPU-util 不代表整段平均。
长输入与更多候选会增加实际开销，4 GiB 进程预算和容量失败回退继续生效。

复测（输出目录必须不存在）：

```bash
bundle exec ruby benchmarks/decision/batch_sweep.rb config/decision/massive.yml \
  data/decision/massive-expanded-control/tokenizer.json \
  data/decision/massive-expanded-control/train.jsonl runs/decision/my-batch-sweep
```

监督与 MLM 使用共同 accumulation 配置；新的 32×1 配置针对此监督训练，不应把它当成原 8×8 MLM 的等价替换。
