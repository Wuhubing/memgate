# memgate

在 LLM 固定不动的 agent harness 里训练一个 memory 准入控制器：每一轮决定哪些 memory 进入模型输入，目标是整个 session 的累计回报。核心问题是这个决策能否当作 bandit，还是必须当作 MDP 做长期规划。

- 执行计划：[plan.md](plan.md)（以下两份原文档的增量与规格）
- 项目文档：[docs/项目文档.md](docs/项目文档.md)
- 工程实现文档：[docs/工程实现文档.md](docs/工程实现文档.md)
- LLM：DeepSeek API（OpenAI 兼容接口，`deepseek-flash`，关闭思考模式），通过 `openai` SDK 调用

## 安装

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,embed]"    # embed = sentence-transformers（MiniLM 嵌入，正式实验必须装）
cp .env.example .env             # 填入 DEEPSEEK_API_KEY，.env 不进 git
pytest                           # 默认跳过 slow；完整检查用 pytest -m slow
```

## 密钥

API key 只放在本地 `.env` 或 shell 环境变量里，代码只读环境变量 `DEEPSEEK_API_KEY`（`scripts/_common.py` 用 python-dotenv 加载 `.env`）。`.env`、`data/`、`runs/`、`results/`、`cache/` 都在 `.gitignore` 里。提交前用 `git diff --cached | grep -i "sk-"` 检查一遍。

## 预算

`configs/llm.yaml` 里 `budget_cny: 60`。每次真实调用的花费按响应里的缓存命中 / 未命中 / 输出 token 记进 `cache/llm_cache.sqlite` 的 spend 表，跨所有脚本累计；到 60 元时 runner 停下（已完成的 session 都保存了，提高上限后重跑同一命令会从断点继续）。`offpeak_only: true` 时高峰时段（北京时间工作日 8–12、14–18、19–22 点）自动等待。价格来自第三方文章，第一次真实调用前请在 DeepSeek 控制台核对一次。

所有真实回答都缓存在同一个 sqlite 里，重跑任何脚本都不会重复花钱。

## 运行顺序（plan §6.2，R1–R10）

在 Mac 上按顺序执行。每一步都可以中断后重跑；先加 `--fake` 可以用离线的 FakeLLM 把整条流水线走一遍（数字没有意义，只检查能跑通）。

```bash
# R1 世界和固定脚本（MiniLM 嵌入）
python scripts/gen_world.py --out data/world_s0

# R2 / E0 预实验：注入的 memory 会不会持续影响后续轮次（对照组 = 不注入）→ 看 G0
python scripts/run_pilot.py --world data/world_s0 --out results/e0

# R3 日志：两种模式各 1500 个 session（70% 均匀随机 + 30% 阈值策略），外加验证用户日志
python scripts/collect_logs.py --world data/world_s0 --mode persistent --out data/logs
python scripts/collect_logs.py --world data/world_s0 --mode ephemeral  --out data/logs
python scripts/collect_logs.py --world data/world_s0 --mode persistent --split val --out data/logs

# R4 拟合行为模型（模拟器）→ runs/sim/report.json、校准图
python scripts/fit_sim.py --logs data/logs --world data/world_s0 --out runs/sim

# R6 / E3 小实例上的精确 DP：Δ、Δ_amort、R_behavior 下的 Δ（按 (mode, T) 多进程）
python scripts/solve_dp.py --sim runs/sim --out runs/e3

# R7 在模拟器上训练 / 调参全部方法
python scripts/train.py --sweep configs/exp/e1_main.yaml configs/exp/e2_gamma.yaml \
    configs/exp/e4_ablation.yaml configs/exp/e5_ephemeral.yaml configs/exp/e_behavior.yaml \
    configs/exp/e3_full.yaml

# R8 模拟器上用测试用户评估 → results/sim_eval.parquet
python scripts/evaluate.py --backend sim

# R9 / E6 真实 LLM 上验证方法排名（每个方法取验证集最好的种子，300 个相同的测试 session）
python scripts/evaluate.py --backend live --select configs/exp/e6_live.yaml

# R10 图表 → figures/（缺的输入会跳过并提示）
python scripts/make_figures.py
```

E3 的过时比例扫描：用 `gen_world.py --p-stale 0.2 --out data/world_p02 --seed 0` 等生成新世界（行为模型可以复用 `runs/sim`，特征与具体世界无关），复制 `configs/exp/e3_full.yaml`，改 `name`（保持 `e3_full` 前缀）和 `world`，再跑 R7、R8。

## 输出

| 文件 | 内容 |
| --- | --- |
| `results/e0/summary.csv`、`g0.json` | E0 各条件各轮的提及率 / 旧值使用率及相对"不注入"的差值 |
| `runs/sim/report.json` | 行为模型的 held-out 对数损失、校准、与 GBT 对比、模拟 vs 真实回放 |
| `runs/e3/delta.csv`、`contrasts.csv` | 小实例上各策略的精确价值，PRIMARY_CONTRASTS 中的 DP 对比 |
| `runs/<sweep>/<run>/run.json` | 每个训练运行的配置、验证回报、训练时长 |
| `results/sim_eval.parquet`、`live_eval.parquet` | 每个 (运行, 测试 session) 一行 |
| `figures/` | 图 1–5、表 1、案例图、`summary.json`（主结论只用其中的 PRIMARY_CONTRASTS） |

## 代码结构

```
memgate/
  config.py schema.py text.py experiment.py
  world/      值池、模板、别名索引、世界与 session 生成器
  retrieval/  嵌入（MiniLM / 测试用 hash）、BM25 + 余弦 RRF 检索
  env/        SessionState、50 维特征、sim / live 后端、gym 环境
  harness/    打分器、prompt、DeepSeek 客户端（缓存/重试/预算/优惠时段）、FakeLLM、runner
  sim/        行为模型特征、数据集、逻辑回归行为模型
  planning/   小实例与精确 DP
  policies/   启发式、LinUCB、模型贪心 / 摊销贪心、SB3 DQN / PPO
  analysis/   统计（bootstrap、配对差、PRIMARY_CONTRASTS）、作图
scripts/      R1–R10 入口
configs/      llm / world / reward，exp/ 下是各实验的 sweep
tests/        plan §5 的全部测试
```
