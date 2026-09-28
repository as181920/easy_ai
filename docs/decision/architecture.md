# EasyAI::Decision 的算法与边界

## 为什么采用这个网络

任务是比较候选，不需要逐 token 生成答案。实现采用共享双向 Transformer encoder，加候选到状态的 cross-attention。双向注意力能同时读取句子的两侧；候选交互保留状态的 token 级信息；共享打分器允许每次传入不同数量、不同文本的选项。候选 ID 只用于关联结果，不进入模型。

这不是固定标签分类头，也没有加载 mmBERT 或 Qwen 的预训练权重。BERT 类 MLM 是这里的预训练目标，Transformer 是网络构件，mmBERT 是具体已有模型，三者不能直接当作互斥的算法选择。纯双塔点积更便宜，但表达细粒度条件关系有限；把状态与每个候选全部拼接的 cross-encoder 交互更完整，却要为每个候选重复编码状态。当前结构为状态复用与交互能力取折中，尚无实验证据表明它优于所有替代结构。[Poly-encoders 论文](https://arxiv.org/abs/1905.01969)也研究了类似的编码复用与交互成本取舍；本项目没有复现其具体结构。

```text
state: [CLS] state [EOS]           option: [CLS] question [SEP] option [EOS]
           |                                          |
           +---- shared token embedding + positions --+
           |                                          |
           +---- shared N x PreLN encoder block ------+
           |                                          |
       memory [L,D]                  option queries [M,D]
           |                                          |
           +-----------------> cross-attention + FFN --+
                                                      |
                                          mask-aware mean pooling
                                                      |
                                            shared Linear(D,1)
                                                      |
                                 all K scores -> softmax(score / T)
```

默认正弦位置编码；GELU FFN；padding 在 attention key 和 pooling 中被遮蔽；变长候选的虚拟项最终置为负无穷。MLM 复用 embedding 作为输出矩阵，仅计算被选中位置的词表 logits，避免构造所有 token 的完整词表输出。

关系实验新增可配置对照：`position_encoding: rotary` 在 encoder self-attention 的 Q/K 上应用 RoPE，不增加参数，并替代输入正弦位置向量；`pooling: candidate` 仅汇集候选内容；`encoding_mode: joint` 拼接状态/问题/候选做联合编码，省去 cross-attention 与 matching 头、禁用状态缓存。默认仍为 `sinusoidal / all / separate`，保持旧 checkpoint 行为。[关系实验](relations.md)记录为何引入这些选项和实际对照结果；不要把新增开关本身当成语义质量保证。

默认模型：vocab 上限 32,000，hidden 256，encoder 4 层，4 heads，FFN 768，interaction 1 层，共 **11,484,929 参数**。词表训练可能因语料不足而提前结束；embedding 按配置容量分配。状态预算 256 tokens、问题和候选合计 64 tokens，均包含特殊 token。正式配置超过预算报错，smoke 配置显式截断并报告计数。截断按 token 前缀，可能截掉否定条件；业务输入应优先保持在预算内。

## 概率与结构化输出

```text
s_i = score(state, question, option_i)
p_i = exp((s_i - max(s)) / T) / sum_j exp((s_j - max(s)) / T)
```

没有 top-k、top-p、beam search 或文本 sampler。给定固定权重和输入，关闭 dropout 后执行确定性的数值计算；CPU/CUDA 内核可能带来小量浮点差异，不承诺跨设备逐 bit 一致。

这些数值是“在所给候选集合中”的归一化概率，不是候选各自成立的独立概率。增删候选会改变分母；多个候选可以同时成立的任务需要另行定义多标签目标。布尔判断可用 true/false 两个候选。若可能全部不成立，需要显式加入 none/other 并准备相应训练数据，不能把最大概率自动解释为事实置信度。

Ruby 组装 `probabilities` Hash，用 `JSON.generate` 输出。输入要求至少两个选项、唯一非空 ID、非空 UTF-8 文本；ID 支持字符串、整数及有限浮点数，统一转成字符串后检查唯一性，训练 target 采用相同规则。输出不会出现语言模型补充说明或无法闭合的 JSON。结构合法与模型判断正确是两个独立问题。

温度 T 在独立 calibration split 上通过有界一维优化拟合，目标是降低 NLL，搜索范围 [0.05,20]，与 T=1 比较。正温度保持 argmax 不变；`calibrated: true` 只表示拟合过温度。评估报告 accuracy、NLL、Brier、ECE 和按语言分组的指标。优化 NLL 不保证 ECE 改善，也不保证换业务域后依然可靠。

## 训练与数据

从随机参数开始：

```text
public train text -> byte BPE -> masked-language-model pretraining
                                      |
                            shared encoder weights
                                      |
                      state/question/options supervised CE
                                      |
                    validation selection -> separate calibration
                                      |
                                held-out test
```

MLM 对约 15% 的内容 token 选择位置，执行 80% mask / 10% 随机 token / 10% 保留。长文档随机窗口，validation 使用固定种子；验证 batch 大小改变会改变具体 mask。候选训练使用交叉熵，每步先均匀抽语言、再在该语言中有放回抽样；`steps` 表示优化器更新次数，不表示 epoch。

梯度累积、范数裁剪、AdamW、线性 warmup 后恒定学习率。学习率和衰减可配置。优化器由 Ruby 实现并与 LibTorch AdamW 做过数值对照，状态按参数名保存。权重、矩、更新步数、配置、数据 SHA256、分组、分词器和增长状态写入 checkpoint。保存采用版本目录与原子 latest 指针，读取校验文件摘要。checkpoint 是可信本地训练产物格式，不是面向不可信上传文件的解析协议。

MLM 对 interaction 和候选头不提供监督，后续候选训练会学习这部分。从预训练 `--init` 开始新任务时优化器和任务步数重置；`--resume` 恢复优化器、采样步数与结构。CPU 固定配置的带 dropout 续训已验证与连续训练一致。数据内容或验证集变化会阻止直接 resume。转入新任务仍保留之前使用过的数据分组，避免 calibration/test 混入先前训练集。

第一版不需要 RL：已有明确候选标签时，交叉熵是直接可用的监督目标。未来多步流程存在延迟回报、动作成本和状态转移时，再在上层编排器研究 RL；当前接口可作为它的动作概率组件。

## 缓存与设备

双向 encoder 中，追加 token 会改变已有 token 的表示，不能照搬生成式 LLM 的增量前缀 KV cache。这里缓存完整 state 的最终 hidden memory。同一 Predictor 实例中，完全相同的编码 state 可以跨问题复用，LRU 默认 16 项；候选分块打分后统一 softmax。候选重排和分块大小不会改变数学上的结果。

另有 CPU token LRU，最多 1024 项，只缓存不超过 4096 tokens 的文本。Predictor 内部串行处理同实例请求；其模型在使用期间应视为只读，更新权重后创建新 Predictor，避免复用旧 hidden memory。当前没有跨进程缓存、量化、融合 attention 或 KV 增量解码。

默认 `auto` 尝试 CUDA，6 GiB 显卡使用 4096 MiB **软预算**。通过 nvidia-smi 采样当前进程占用；这不是精确瞬时峰值，也不能预留 GPU 内存。nvidia-smi 缺失时依赖 CUDA 实际容量错误回退。训练先写 CPU checkpoint，发生容量错误后恢复已提交状态、缩小 microbatch 并增加累积次数，仍不足则切到 CPU。推理先缩小候选 chunk，再切 CPU。非容量类程序错误不会被静默吞掉。

只实现并验证 FP32；没有假装启用 AMP。Torch.rb 当前接口没有释放 CUDA allocator 缓存的入口，切回 CPU 后进程可能仍保留缓存显存，退出进程才能完全归还。输入长度、候选数和 batch 才是激活显存的主要变量；不能仅凭参数量断言任意请求均可跑下。

## 动态增长

支持两种有明确参数映射的增长方式：

- `add_block`：新增 PreLN residual block，将 attention 输出投影与 FFN 输出投影置零，使新增块初始为恒等映射。
- `widen_ffn`：只扩大某 encoder block 的 FFN 中间维度，保留旧权重；新增输出列置零，使原函数连续，同时允许后续学习。

迁移已有 AdamW moments，新增参数或新切片 moments 为零。扩宽张量保留该张量已有 step，因此新切片沿用旧偏差修正步数；这是当前明确的近似。任意 hidden size 改大还牵涉 heads、LayerNorm 和 embedding，本版不支持这一类无约束变形。相关方法背景见 [Net2Net](https://arxiv.org/abs/1511.05641)。

自动增长默认关闭。启用后同时观察训练和验证 loss 平台，再创建一个可回滚的增长试验；降低短期学习率，若规定次数内验证 NLL 没有改善则回滚，显存不合适也回滚。训练预算结束时尚未接受的试验不会成为最终模型。回滚保留已经消耗的 step/example 计数；优化器矩恢复至增长前。默认上限 6 层、两次试验，也可改为 FFN 扩宽。平台期不等于容量不足，增长仍需验证，不能无限加层。

结构或权重变化后需重新校准。手动 `grow` 生成新训练产物并使原温度失效；自动增长的试验事件保存在元数据中。

## 状态不敏感的排查开关

旧配置的 `position_scale: 1.0`、`embedding_norm: false`、`score_mode: linear` 仍作为兼容默认值保留，旧 checkpoint 不会被静默解释成新结构。
可显式设置 `position_scale: 0.02` 与 `embedding_norm: true`，在 embedding 加位置后进行 LayerNorm；hidden 256 时新增 512 个参数。
仅调整位置尺度的对照没有解决状态不敏感，不能把它单独称为修复。

实验性 `score_mode: matching` 保留共享 encoder 和 cross-attention，在池化后显式构造状态与候选的联合特征：

```text
state memory -------- masked mean ------ s [D] ----+
                                                  |
question + option --> encoder --> interaction --> q [D]
                                                  |
                             concat(q, s, q*s, |q-s|) [4D]
                                                  |
                                      Linear(4D,D) + GELU
                                                  |
                                         Linear(D,1)
                                                  |
                                all candidates -> softmax
```

`q*s` 是逐元素乘积，不是概率乘积。新增投影在 hidden 256 时含 262,400 个参数，联合输入归一化后总参数为 11,747,841（vocab 上限 32,000）。
它让打分头直接学习两种表示之间的关系；并不强制网络依赖 state，也不保证理解否定句，仍须用状态消融和独立任务数据验收。
缓存继续只保存 state memory，候选分块与全局 softmax 不变。ID 不进入神经网络。
