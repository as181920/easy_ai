# 02 · 训练：SGD → AdamW

已实现可运行的最小教学实验。先修：01_basic_nn。 下一章：[03_diagnostics](../03_diagnostics/README.md)。

复用 01 的 MLP 思路，换成 96 个连续象限分类样本和独立验证集；全部优化器使用相同结构、划分、初始化与 full-batch 更新预算。XOR 的四点真值表适合手算，不适合比较泛化正则化。

训练顺序：forward → mean loss → zero_grad → backward → 裁剪/诊断 → optimizer.step → eval/no_grad 验证。记录的 train loss 是本次更新前的训练模式值，validation loss 是更新后评估模式值，dropout 与时点不同，不要求两条曲线逐点相等。

| 方法 | 本章实现 |
| --- | --- |
| SGD / Momentum | `θ-=ηg`；`v=μv+g, θ-=ηv`；普通 L2 加进梯度 |
| Adam | 一阶/二阶矩与偏差修正；epsilon 在开方之后 |
| AdamW | `θ=(1-ηλ)θ-η m̂/(sqrt(v̂)+ε)`；衰减不进入矩估计 |
| Dropout / decay | 独立消融；train/eval 模式；不保证每次提升 |
| Schedule / warmup | warmup 后 cosine，01 步长与最终端点可手算 |
| Accumulation | 每个 microbatch mean loss 按真实样本/token 数加权 |
| EarlyStopping | 独立验证集、patience/min_delta、恢复最佳模型状态 |
| GradScaler | 显式 loss scaling、梯度反缩放、非有限梯度跳过更新 |

[标量参考](../lib/easy_ai_learning/training/scalar_optimizer.rb) → [张量优化器](../lib/easy_ai_learning/training/optimizer.rb) → [训练循环](../lib/easy_ai_learning/training/loop.rb)。Torch.rb 0.23 的优化器状态接口未实现，本章自行保存矩、velocity、逐参数 step；模型状态另外保存。GradScaler 是显式数值教学，**没有提供自动 autocast/完整 AMP 环境**。

```ruby
loop = EasyAILearning::Training::Loop.new(model, kind: :adamw, lr: 0.02)
loop.run(steps: 60, validation: validation_loss) { training_loss.call }
state = loop.state_dict # 可 JSON 序列化；模型另存 state_dict
```

本章 full-batch 循环按 `seed+update_index` 重置 Torch 随机种子；恢复模型、训练状态及相同数据/objective，并继续使用原 total_steps，可精确重现 dropout/corruption 的下一次更新。该约定只覆盖这些教学循环，不自动恢复 RL 环境、replay、历史文本 batcher 或任意外部随机状态。EarlyStopping 恢复的是最佳推理模型；结束时的 optimizer 状态未回滚，不能与最佳模型拼成“同一时刻”的恢复 checkpoint。

公平优化器比较还应分别搜索学习率并用多种子。这里只是固定预算教学对照。公式依据：[AdamW](https://arxiv.org/abs/1711.05101)、[Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html)。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/02_training/data.rb
bundle exec ruby learning/02_training/train.rb --steps 60
bundle exec ruby learning/02_training/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/02_training/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/adamw-loss.svg)

[实验代码](../lib/easy_ai_learning/training/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/training_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
