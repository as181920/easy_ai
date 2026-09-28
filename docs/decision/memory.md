# 验证与推理的张量生命周期

2026-09-28：长文本候选训练第一次进入 validation 时，进程显存从训练期间的可用范围不断增长到约 4.95 GiB，触发 OOM 和 CPU 回退。CPU 回退是容量不足时的容错，不能作为这种累积问题的解决办法。

## 原因与修复

Torch.rb 的 Ruby 对象持有 LibTorch 原生张量。Ruby GC 并不知道这些包装对象背后有多少 CPU/CUDA 内存；包装对象本身很小，自动回收可能远晚于显存耗尽。`Torch.no_grad` 仅关闭梯度记录，不会销毁中间张量。

验证循环现在每个 batch 都进入独立方法，只返回 Ruby 数值（loss 与样本数/被遮盖 token 数），离开方法后明确运行 GC。验证结束或异常退出也执行回收。MLM 的 loss 仍按被遮盖 token 数加权，固定掩码保证多轮验证可比较。

```text
validation batch
  -> temporary input / attention / logits tensors
  -> loss.item + count (Ruby numbers)
  -> return from batch method
  -> GC: release unreachable tensor wrappers
  -> LibTorch releases/reuses their storage
  -> next batch
```

长期复用的 Predictor 同样在每个候选 chunk 返回 Ruby 分数后回收临时张量，请求结束时再次清理，只保留容量受限的状态缓存。`Evaluator` 和诊断工具不再是唯一可能触发回收的位置，直接调用 `predictor.probabilities` 也受保护。

验证之前会保存可续训 checkpoint，避免验证异常丢失已完成的更新。每次验证在 `choice/metrics.jsonl` 或 `mlm/metrics.jsonl` 中记录：

- `gpu_process_mib_before_validation`
- `gpu_process_mib_after_validation`

CPU 下这两个字段为 `null`。数值来自 `nvidia-smi` 的当前进程占用，包含驱动和分配器缓存，并非全部为活跃张量。正常的预热增长之后应该稳定，不要求回到零。当前 Torch.rb 0.23.0 的 CUDA 绑定没有暴露活跃/保留显存的细分统计，因此这里不冒充 allocator 内部测量。

有上述记录的 run 可用 `bundle exec ruby bin/easy-ai report --input RUN目录` 生成 `report/memory.png`、`memory.svg`，HTML 报告会展示验证前后的显存曲线。旧运行或 CPU 运行缺少有效测量时不绘制，不把缺失值画成零。

## 本机回归结果

使用 `semantic.yml` 的 6,627,841 参数、FP32、batch 16；先执行一次训练更新，保留梯度与 Adam 状态，再连续跑 **5 次完整验证**。候选验证 1015 行，MLM 验证来自同一组独立 validation 文本。

| 路径 | 首轮显存，前 → 后 | 第 3–5 轮显存 | 第 3–5 轮 RSS | 第 3–5 轮活跃 Ruby Tensor 对象 |
| --- | ---: | ---: | ---: | ---: |
| 候选判断 | 1122 → 1178 MiB | 1178 MiB | 1267.12 MiB | 267 |
| MLM | 1178 → 1178 MiB | 1178 MiB | 1287.68 MiB | 227 |

两条路径都保持 CUDA，稳定期这三项指标的最大值减最小值均为零。MLM 在同一进程随后执行，因此包含前一阶段留下的分配器缓存。另一次监督训练已经完成 1000 次更新和 10 次验证，没有回退 CPU。

随后 `semantic-mlm-v1` 完成从零 MLM 1000 步 + 监督 1000 步：MLM 的 10 个验证点均为 1220 → 1220 MiB；监督首次验证 1590 → 1602 MiB，之后 9 个验证点均为 1602 → 1602 MiB。完整训练和固定权重压力测试都没有出现持续累积。

这是指定配置和数据下的回归证据，不代表任意长度、候选数都不可能超出硬件容量。单个 batch 本身过大仍应减小 microbatch 或候选 chunk。

复现命令（需要本地语义数据和可访问的 NVIDIA GPU）：

```bash
bundle exec ruby benchmarks/decision/validation_memory.rb > /tmp/easy-ai-validation-memory.json
```

脚本对两轮预热后的三轮测量设定上界：进程显存波动 96 MiB、RSS 波动 128 MiB、Ruby Tensor 数波动 16；超出、无法测量、非有限 loss 或回退 CPU 都失败。还会每 8 个 batch 采样一次；此处采样的是 batch 间占用，不是精确的 batch 内峰值。

普通测试无需 GPU：

```bash
bundle exec ruby -Itest test/easy_ai/decision/memory_test.rb
```

测试关闭自动 GC 来模拟原生内存不可见的压力，检查重复 choice/MLM 验证和长期 Predictor 的 Tensor 数有界；验证 loss 可重复且不产生参数梯度。修复位于正式训练/推理路径，不依赖 benchmark 代为回收。
