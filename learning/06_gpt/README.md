# GPT

The existing decoder-only GPT text-training example is `train.rb`. Model, trainer, configuration and batch/text helpers are in `../lib/easy_ai_learning/gpt/`. Run from the repository root: `bundle exec ruby learning/06_gpt/train.rb --data data/learning/song.txt --tokenizer byte --iters 200 --device cpu`. The local corpus is ignored and must exist. The original repository learning illustrations are preserved.

See [the learning progression](../README.md).

The roadmap continues with [reinforcement learning](../07_rl/README.md). RL changes the training objective and interaction loop; it can train an MLP or a GPT policy and does not require a text model as its starting point.
