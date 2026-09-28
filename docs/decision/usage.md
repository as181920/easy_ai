# 使用指南

以下命令从仓库根目录执行。依赖安装和 CUDA 构建参见根 README。`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 适合测试和小 batch，CPU 正式运行可实测后调整线程数。CLI 的最终结果是 JSON，训练日志默认写入 `log/`。

## 一键训练与效果图

当前从零语义实验（本地已准备中英公开监督数据；缺失时下载并准备）：

```bash
bundle exec ruby bin/easy-ai semantic-pipeline

# 随机初始化直接监督，作为 MLM 的对照
bundle exec ruby bin/easy-ai semantic-pipeline --mlm-steps 0
```

默认使用 `semantic.yml`，MLM/候选各 1000 步；独立输出 run，不覆盖旧意图权重。详细数据定义、许可和质量限制见[语义训练说明](semantics.md)。

改进后的意图匹配实验（从随机权重做监督训练）：

```bash
bundle exec ruby bin/easy-ai pipeline --config config/decision/massive.yml --train-limit 2000 --backend native --vocab-size 8000 --mlm-steps 0 --choice-steps 1000 --eval-every 100
```

`--train-limit` 只调整每语言的训练行数，其他 split 仍由 `--limit` 控制，默认各 200 行，方便保持相同验证集做对照。
原来的小数据 MLM 教学流程保留如下：

```bash
bundle exec ruby bin/easy-ai pipeline
```

默认从 `data/decision/downloads/massive-1.1.tar.gz` 准备五语言数据，每种语言每个 split 最多 200 条，每例 8 个候选。
本地原始包不存在时才下载。自训 Ruby BPE，使用 `small.yml` 做 MLM 100 步、候选监督训练 300 步，随后校准、测试并输出效果图。
该数据规模用于本地实验，增加更新次数不能替代扩大独立样本数量。

每个阶段使用独立 Ruby 子进程，结束后释放该进程的 GPU 资源。原始小数据 pipeline 默认每 10 步验证一次（根据训练步数与配置计算），可用 `--eval-every` 修改；定期 checkpoint 间隔由模型配置决定，GPU 每次验证前还会保存已完成的更新，以便验证异常后恢复。
不修改原来的 YAML 文件，本次基础配置保存在 run 目录中的 `config.yml`，各阶段覆盖的步数见 `summary.json` 和各 checkpoint 的 metadata。

```bash
# 快速检查全部流程
bundle exec ruby bin/easy-ai pipeline --config config/decision/smoke.yml --limit 12 --mlm-steps 10 --choice-steps 20

# 指定新输出目录；在终端中查看实时进度
bundle exec ruby bin/easy-ai pipeline --output runs/decision/my-experiment

# 使用所选语言的完整 MASSIVE 数据，增加训练预算
bundle exec ruby bin/easy-ai pipeline --full-data --backend native --mlm-steps 1000 --choice-steps 3000 --eval-every 100

# 复用已经准备好的六个 JSONL 文件与自训 tokenizer
bundle exec ruby bin/easy-ai pipeline --data data/decision/massive-validation --tokenizer data/decision/massive-smoke/tokenizer-ruby.json
```

输出目录必须不存在；默认按时间和随机后缀生成，避免覆盖历史实验。`--data` 在 pipeline 中是数据目录，须包含 train、validation、calibration、test、corpus、corpus-validation 六个 `.jsonl`。
不传 `--tokenizer` 时仅使用本次 train 数据训练 tokenizer；`--backend native` 可加速。使用别人准备的 tokenizer 时，调用方需要确认其没有读取 calibration/test 数据。
超长输入会在训练前按配置检测；正式配置默认报错，可明确调整长度预算或截断策略后重新运行。

进度输出到 stderr，最终摘要是 stdout 的 JSON。例如：

```text
[mlm]    [==========..........] 50/100  50.0% loss=8.1234 val@40=8.4567 device=cuda elapsed=20s ETA~20s
[choice] [==========..........] 150/300 50.0% loss=1.4321 val@140=1.5678 device=cuda elapsed=70s ETA~70s
```

上述数值仅演示格式；`val@40` 明确表示最近完成的第 40 步验证。ETA 按已完成更新的平均时间估计，不包含后续校准、test 和绘图。
另一个终端可以 `tail -f runs/decision/<run>/pipeline.log` 看进度，或查看 `train.log` 中每次更新与验证的完整记录。
`training.jsonl` 记录逐步 loss；`metrics.jsonl` 记录验证点，报告保留同一步重试的最后一次观测。

结束后控制台直接展示 Unicode loss 曲线，并打印 `report/index.html` 路径。用浏览器打开可查看：

- `loss.png` / `loss.svg`：MLM 与候选任务分开绘制训练、验证曲线，不做平滑。
- `evaluation.png` / `evaluation.svg`：calibration 上温度拟合前后指标、test 上的置信度与准确率关系。
- `memory.png` / `memory.svg`：有 GPU 测量的运行展示验证前后进程显存；旧记录或 CPU 运行不伪造测量值。
- 按语言的 accuracy、NLL、Brier、ECE 表格，以及样本行数和独立分组数量。

绘图依赖 gnuplot（本机已经安装），无需 Python；程序会在训练前检查依赖。图像和 HTML 可独立保存，HTML 连同 report 目录一起分享即可浏览图表。
README 中的历史 learning 曲线继续保留，learning 脚本原有的 UnicodePlot 输出也继续有效。

可在训练结束或中断后重新绘制已有 trace：

```bash
bundle exec ruby bin/easy-ai report --input runs/decision/my-experiment
```

完成后再绘制可得到完整评估；中断后若已有 training trace，则可以绘制部分曲线。旧版仅记录 validation 的产物没有逐步训练 trace，无法凭空补出历史曲线。
pipeline 不会自动覆盖或续跑已有目录；失败状态及已完成阶段保存在 `summary.json`。续训使用阶段级 `pretrain/train --resume`，再执行校准和评估。普通阶段命令也可加 `--progress` 显示进度。

## 数据下载和准备

第一套真实数据采用 [Amazon MASSIVE 1.1](https://github.com/alexa/massive)，公开多语言意图数据，许可证 CC-BY-4.0。默认选取中、英、日、西、阿五种语言；原数据覆盖 52 种语言，可通过 `--locales` 选择其他 locale。

```bash
bundle exec ruby bin/easy-ai download \
  --output data/decision/downloads/massive-1.1.tar.gz

bundle exec ruby bin/easy-ai prepare \
  --archive data/decision/downloads/massive-1.1.tar.gz \
  --output data/decision/demo \
  --locales zh-CN,en-US,ja-JP,es-ES,ar-SA \
  --candidates 4 --limit 40
```

若需要代理，可在下载命令前设置 `https_proxy=http://127.0.0.1:20122 http_proxy=http://127.0.0.1:20122`，或用本机已有的 proxychains 包装该命令。下载器使用 Faraday 的 Net::HTTP adapter，流式写 `.part`，完成后移动目标文件；可用 `--sha256` 指定预期摘要。不会覆盖已有文件；输出目录也应使用新路径。

`--limit` 是每种语言、每个 split 的上限。去掉它使用所选语言的完整数据。adapter 保留官方 train/test，将 dev 按原始样本 ID 分成 validation/calibration；不同语言的同一原始样本归属同一 split。输出：

```text
data/decision/demo/
|-- train.jsonl
|-- validation.jsonl
|-- calibration.jsonl
|-- test.jsonl
|-- corpus.jsonl              # 仅 train 的原始文本，给 MLM
|-- corpus-validation.jsonl   # 仅 validation 的文本
|-- manifest.json             # 来源、摘要、候选与切分策略
`-- LICENSE
```

每条样本包含目标候选加抽样负例，所以四候选测试不等于完整 60 类 MASSIVE 基准。默认问题为 `Intent?`，候选描述来自英文 intent 名。中文等输入在此学习跨语言意图映射；若要中文问题/中文候选或其他业务任务，需要对应训练数据。`--descriptions path.json` 可提供 `{ "intent_id": "描述" }`，必须覆盖训练标签。

通用监督 JSONL 每行格式（ID 是输出键，模型只读 text）：

```json
{"id":"sample-1","group_id":"source-1","language":"zh-CN","state":"订单被扣了两次款","question":"交给哪个部门？","options":[{"id":"billing","text":"账单与退款"},{"id":"shipping","text":"物流配送"}],"target":"billing"}
```

自备数据时，相同文档、改写和翻译必须共享 `group_id`，再分配到互斥的 train/validation/calibration/test。工具会拒绝已知分组交叉，但无法识别用不同 ID 隐藏的语义重复。勿把 calibration/test 用于训练分词器。

额外的公开文本可先导出为本地 JSONL（每行至少 `text`，可有 `id`、`language`、`source`），再准备 MLM 语料：

```bash
bundle exec ruby bin/easy-ai prepare-corpus \
  --input data/public-text.jsonl --output data/decision/public-corpus --language zh-CN
```

此命令按规范化文本 SHA256 去除完全重复并以 95/5 拆分 train/validation，写入来源摘要。它不做语义近重复检测或平行翻译分组，也不自动下载大型网页语料。保留公开数据原许可证与来源，并为领域泛化补充有代表性的语料。

## 自训 tokenizer 与模型

先用 Ruby BPE 完成可阅读的端到端流程：

```bash
bundle exec ruby bin/easy-ai tokenizer \
  --data data/decision/demo/train.jsonl --backend ruby --vocab-size 4096 \
  --output data/decision/demo/tokenizer.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bundle exec ruby bin/easy-ai pretrain \
  --config config/decision/smoke.yml --tokenizer data/decision/demo/tokenizer.json \
  --data data/decision/demo/corpus.jsonl \
  --validation data/decision/demo/corpus-validation.jsonl \
  --output runs/decision/demo-mlm

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bundle exec ruby bin/easy-ai train \
  --init runs/decision/demo-mlm --data data/decision/demo/train.jsonl \
  --validation data/decision/demo/validation.jsonl --steps 30 \
  --output runs/decision/demo-choice
```

`smoke.yml` 使用约 15.7 万参数，意在快速发现接口问题。正式起点改用 `small.yml`，tokenizer 的 `--vocab-size` 可改为 32000，同时扩大训练数据和训练预算。小语料不一定学满词表。不要把 smoke 的十几步训练视作语言能力训练完成。

两个 backend 都从给定文本自行训练，不下载模型 tokenizer：

- `ruby`：纯 Ruby byte BPE，保留空白、任意 UTF-8 字节覆盖；训练以增量 pair 索引和 heap 更新，适合学习与小规模实验。
- `native`：`tokenizers` gem 的 Rust BPE，用于更大语料。预分词和合并规则与 Ruby 版本不同，必须分别训练、保存并绑定对应权重。当前 backend 拒绝包含 `[PAD]`、`[UNK]`、`[CLS]`、`[SEP]`、`[MASK]`、`[EOS]` 字面量的用户文本，避免 gem 的特殊 token 解析损坏原文往返；有这类输入时选 Ruby backend。

分词训练保存每种唯一 chunk 的统计；它不是常量内存的大规模流式 BPE 训练器。大型语料先按语言均衡抽取词表训练样本，用正式训练吞吐实测决定 backend。JSONL 训练集使用行偏移索引，验证遍历按行流式读取，但索引与采样分组仍占 CPU 内存。

## 续训、校准、评估与推理

```bash
bundle exec ruby bin/easy-ai train \
  --resume runs/decision/demo-choice --steps 40 \
  --data data/decision/demo/train.jsonl \
  --validation data/decision/demo/validation.jsonl \
  --output runs/decision/demo-choice

bundle exec ruby bin/easy-ai calibrate \
  --checkpoint runs/decision/demo-choice --data data/decision/demo/calibration.jsonl \
  --output runs/decision/demo-calibrated

bundle exec ruby bin/easy-ai evaluate \
  --checkpoint runs/decision/demo-calibrated --data data/decision/demo/test.jsonl

bundle exec ruby bin/easy-ai predict \
  --checkpoint runs/decision/demo-calibrated --input examples/decision/request.json
```

命令均可用 `--device cpu` 强制 CPU。`auto` 和 `cuda` 均在 CUDA 不可用或容量不足时按策略回退；`cuda` 不是禁止回退的严格模式。`--steps` 是目标总步数；小于等于 checkpoint 的步数不会额外训练。resume 必须保持相同训练和验证文件，配置与 tokenizer 从 checkpoint 恢复，只允许修改总步数和设备。

校准输出是独立推理产物，不含可续训优化器；继续训练请 resume 校准前训练产物，或明确用 `--init` 开始新训练阶段。`evaluate` 检查 test 不与训练、验证、校准分组重叠。组级防泄漏假定输入如实设置 group_id。

`predict` 未提供 `--input` 时从 stdin 读取 JSON，格式见 [request.json](../../examples/decision/request.json)。结果形状如下，数值仅为格式示例：

```json
{"probabilities":{"alarm_set":0.5,"weather_query":0.3,"music_play":0.2},"calibrated":false,"temperature":1.0,"device":"cpu","truncated_segments":0}
```

`truncated_segments` 是该请求内 collator 执行截断的次数，分块和重试可能重复计数，不是去重后的字段数量。

Ruby 公共 API：

```ruby
require "easy_ai"

predictor = EasyAI::Decision::Predictor.load("runs/decision/demo-calibrated")
result = predictor.probabilities(
  state: "明天七点叫我起床",
  question: "Intent?",
  options: [
    { id: "alarm_set", text: "alarm set" },
    { id: "music_play", text: "music play" }
  ]
)
puts JSON.generate(result)
```

候选 `id` 和训练 `target` 支持字符串、整数及有限浮点数，进入 API 后统一转成字符串，保证 Ruby Hash 与 JSON 的键一致。
例如 `id: 0` 的输出键为 `"0"`，`id: 2.5` 为 `"2.5"`；同一请求不能同时使用 `1` 和 `"1"`，归一化后重复会报错。
ID 只用于标识结果，模型读取的是 `text`，数字 ID 不表示顺序或大小。布尔选项应使用 `id: "true"` / `id: "false"`，不接受 Ruby 布尔值作为 ID。

注意：MASSIVE 训练的是意图路由。`state: "不会迟到了"`、问题 `"会迟到么"`、候选 `"会/不会"` 属于另一种语义判断任务，当前数据没有为它提供充分监督。
输出合法的 JSON、概率和为 1，均不代表已学会该任务；温度校准也不会修复答案排序。
`calibrated: true` 表示加载了校准温度；温度在八候选意图数据上拟合，并不代表任意二选一问答也经过校准验证。

[examples/decision/predict.rb](../../examples/decision/predict.rb) 另示范中文客服候选调用，只展示接口；MASSIVE smoke 模型没有接受客服业务训练。

## 动态增长

手动增长先生成新产物，再继续训练：

```bash
bundle exec ruby bin/easy-ai grow --checkpoint runs/decision/demo-choice \
  --operation add-block --output runs/decision/demo-grown

bundle exec ruby bin/easy-ai train --resume runs/decision/demo-grown --steps 50 \
  --data data/decision/demo/train.jsonl --validation data/decision/demo/validation.jsonl \
  --output runs/decision/demo-grown
```

扩大 FFN 用 `--operation widen-ffn --layer 0 --size 128`（必须大于当前对应层宽度），输出使用新目录。这里 128 针对 smoke 的 64 宽度；small 初始 FFN 为 768。

自动试验在新训练配置中设置：

```yaml
growth:
  enabled: true
  patience: 3
  min_delta: 0.001
  trial_evaluations: 3
  max_layers: 6
  max_trials: 2
  operation: add_block
```

必须提供 validation。每 `eval_every` 步评估一次，平台触发试验；未改善则恢复增长前权重和优化器。启用时设置足够训练预算，让试验有后续验证机会。详细语义见[架构说明](architecture.md)。

## 产物与检查

```text
runs/decision/demo-choice/
|-- latest.json
|-- metrics.jsonl
`-- checkpoints/step-00000040-<suffix>/
    |-- weights.pt
    |-- optimizer.pt       # 有优化器矩时存在
    |-- tokenizer.json
    |-- metadata.json
    `-- manifest.json
```

可传 run 目录或具体 checkpoint 目录。metadata 记录结构、步数、数据分组、SHA256、增长事件与校准信息；不同时运行两个写入相同 run 目录的训练任务。保存前验证输出磁盘空间，checkpoint 会持续积累；本版没有自动删除历史产物，以保留增长回滚依赖。

```bash
bundle exec ruby bin/easy-ai inspect --checkpoint runs/decision/demo-choice
bundle exec rake test
bundle exec rake test:learning
bundle exec rake lint
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bundle exec ruby benchmarks/decision/verify_gpu.rb
bundle exec ruby benchmarks/decision/tokenizers.rb data/decision/demo/train.jsonl 4096
git check-ignore data/decision/demo/train.jsonl runs/decision/demo-choice/latest.json
```

`verify_gpu.rb` 必须在可访问 NVIDIA 设备的环境运行。分词器基准同时报告训练、编码时间和原文往返；真实训练基准的命令见[本机验收](validation.md)。

## 检查模型是否真的使用状态

一键流程现在自动对选中的权重做 validation 状态消融，并写入 `stage-results/diagnostic.json` 和 HTML 报告。也可以单独检查已有权重：

```bash
bundle exec ruby bin/easy-ai diagnose \
  --checkpoint runs/decision/pipeline-showcase/choice \
  --data runs/decision/pipeline-showcase/data/validation.jsonl \
  --reference-data runs/decision/pipeline-showcase/data/train.jsonl \
  --language zh-CN --limit 200 --device cpu
```

它保持问题、候选、目标不变，比较原始、打乱和常量 state，以及训练标签频率基线。默认优先中文，每个原始分组取一条、最多 200 条。
这个检查使用未校准 logits；校准不会改变 argmax。打乱状态也会制造不匹配输入，预测变化本身不是语义正确的证明，应结合原始输入的 NLL、准确率和基线一起看。

每次验证下降时，训练器在 `choice/best/`（或 `mlm/best/`）保存推理/迁移权重，一键流程用选中的权重校准和评估。
`choice/` 的 latest 仍保留最后一次训练的优化器，用来 `--resume`；best 只能预测或 `--init`，不能恢复优化器。
`training.early_stopping_patience` 为 0 时不早停，设为正整数表示连续多少次验证无显著改善后停止，阈值是 `selection_min_delta`。开启早停需要验证集。

可选训练开关：

```yaml
training:
  balance_labels: true
  resample_negatives: true
  early_stopping_patience: 6
  selection_min_delta: 0.001
```

标签平衡先等概率采语言，再等概率采该语言下的 `(question, target)`，最后采样一条训练文本。
动态负例从同语言、同问题的训练候选目录重抽，保留正确候选并打乱位置；验证和测试仍使用固定候选。
这两个选项适用于 MASSIVE 一类共享标签目录的任务。候选 ID 随请求改变含义时，不要启用动态负例；相同 ID 描述冲突会报错。
标签平衡改变训练先验，因此继续使用独立 calibration split 校准，并报告测试分布上的指标。

`pipeline --mlm-steps 0` 可以直接随机初始化后做候选监督训练，用于区分短程 MLM 和监督训练本身的影响。
MLM 是可选训练阶段，短短 100 步不等于已得到通用语言模型。旧的 learning 曲线仍独立保留。
