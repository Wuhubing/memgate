# memgate 执行计划

按天、按步骤排好的执行清单，依据《项目文档》和《工程实现文档》，时间线 2026-10-27 至 12-10。每个阶段最后有一道**关卡**：关卡没过，不进下一阶段。

与原文档的唯一改动：**LLM 用 DeepSeek API（OpenAI 兼容接口），不自部署 Qwen3-8B**。影响见第 0 节。

---

## 0. 换成 DeepSeek 带来的调整

| 项目 | 原文档 | 现在 |
| --- | --- | --- |
| 调用方式 | vLLM + Qwen3-8B，或托管 API | `openai.AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])` |
| 模型名 | Qwen3-8B（关闭 thinking） | `deepseek-chat`（非思考模式）。不要用 `deepseek-reasoner`：思考链会拖慢速度、抬高成本、干扰打分 |
| 前缀缓存 | vLLM prefix caching | DeepSeek 自动做上下文硬盘缓存，同一 session 的历史前缀会命中，不需要额外设置。日志里记下返回的缓存命中 token 数，用来核算成本 |
| 可复现性 | 固定带日期的模型版本 | `deepseek-chat` 是别名，官方升级时会悄悄换底层模型。**每次调用都记录响应里的 `model` 字段和日期**；E0、日志收集、E6 尽量集中在同一段时间跑完；如果中途发现版本变了，在报告局限性里写明 |
| seed 参数 | 传 seed | 不要依赖 seed 让结果可复现。随机性来自 temperature 0.7，可复现靠本地 SQLite 响应缓存 |
| 并发 | 信号量 32 | 从 8 开始，E0 时实测速率限制和报错率后再调 |
| E7 换模型 | 换更强的 LLM | 可选：同一套代码换 `base_url` / 模型名即可（例如另一家 OpenAI 兼容 API） |

**密钥规则（必须遵守）**

1. key 只放在 Mac 本地的 `memgate/.env`（从 `.env.example` 复制），或 `~/.zshrc` 里 `export DEEPSEEK_API_KEY=...`。
2. 代码里只读环境变量：`os.environ["DEEPSEEK_API_KEY"]`，用 `python-dotenv` 在脚本入口 `load_dotenv()`。
3. `.env` 已在 `.gitignore` 里。第一次 `git add` 之前跑 `git status` 确认 `.env` 不在列表里。
4. 每次提交前：`git diff --cached | grep -iE "sk-[a-z0-9]{10,}"`，有输出就停下。
5. 日志、parquet、SQLite 缓存里不写 key（只缓存 messages 和响应）。

**价格**：开工前去 DeepSeek 官网价格页查当前的输入（缓存命中 / 未命中）和输出单价，按"约 67M 输入 + 6M 输出"算一遍总预算，写进 `configs/llm.yaml` 的注释里。E0 跑完后用实测 token 数重新估算一次。

---

## 1. 10/27–10/29：环境搭建 + 预实验 E0（第一个关键节点）

时间最紧的一段，只做 E0 必需的东西，其他一律推后。

**步骤**

1. **仓库初始化**（10/27 上午）
   - 创建空 repo `memgate`，加入 `plan.md`、`README.md`、`.gitignore`、`.env.example`。
   - 本地：`python3.11 -m venv .venv`，写 `pyproject.toml`，依赖先装最少的：`openai python-dotenv pydantic pyyaml pandas pyarrow numpy pytest`。
   - `cp .env.example .env`，填入 key。
2. **连通性测试**（10/27）
   - 写 `scripts/smoke_llm.py`：发一条 system + user 消息，打印回答、`model` 字段、token 用量、缓存命中数。
   - 跑通后再继续。
3. **`schema.py`**（10/27）：照工程文档把 `MemoryRecord`、`Turn`、`SessionScript`、`SessionState`、`TurnLog` 的 dataclass 抄进来。
4. **词表与模板**（10/27–10/28）：`world/pools.py`（虚构实体名、值，全部英文）、`world/templates.py`（提问、闲聊、更新句式）。
   - 每个 value 配别名表，并写一个检查：任意两个 value 的别名不重叠。
5. **生成器**（10/28）：`world/generator.py`，每用户 14 个 key（6 会变 / 4 偏好 / 4 琐事），store 30–45 条，session 默认 12 轮（40% 提问 / 45% 闲聊 / 15% 更新），按 seed 完全确定。
   - 跑 `python scripts/gen_world.py --config configs/world.yaml --seed 0 --out data/world_s0/`。
   - 用户按 60/20/20 划分训练 / 验证 / 测试。
6. **harness 最小版**（10/28）
   - `harness/llm_client.py`：异步客户端，指数退避 + 抖动，最多重试 5 次；SQLite 响应缓存，key = hash(模型名, messages, 采样参数)。
   - `harness/prompt.py`：持久模式（memory 拼在当前 user 消息前，留在历史里）和一次性模式（只加在 system prompt 末尾）。
   - `harness/scorer.py`：小写、去标点、合并空白后子串匹配，算答对 / 旧值 / 无关提及 / token 四项。默认系数 δ=1.0、α=0.3、β=1.5、λ=0.002。
   - E0 不需要检索器：四种条件下直接指定注入什么。
7. **打分器单元测试**（10/28）：手写约 30 条回答覆盖全部分支，`pytest` 全过再跑 E0。
8. **跑 E0**（10/29）：`scripts/run_pilot.py`，四种条件（不注入 / 注入正确 / 注入过时 / 注入无关）× 50 session，持久模式。
   - 先跑 5 个 session 人工看一遍回答，确认打分没明显错，再跑全部。
   - 输出 `runs/e0/summary.csv`：注入后第 1–5 轮的重复提及率、用户纠正后的旧值使用率。

**关卡 G0（10/29 晚）**：无关 memory 后续轮次重复提及率 ≥ 10%，**或**纠正后旧值使用率 ≥ 10%。

- 通过 → 进入第 2 节。
- 没通过 → 按顺序试：去掉 system prompt 里"只在相关时使用"那句；让过时信息更隐蔽；换更弱的模型。每次只改一处，重跑 E0 的子集。仍然不过，也继续往下做（token 成本和复用这两个机制仍然产生长期结构），并在 proposal 里写明"模型很稳健"这个发现。

---

## 2. 10/30：提交 proposal

1. 根据 E0 结果确定默认参数（session 长度、过时比例、system prompt 版本）。
2. 写 1 页 proposal：研究问题、H1–H4、MDP 定义、方法表、E0 的初步结果（附图 1 草图）、换用 DeepSeek 的说明。
3. 提交。

**关卡 G1**：proposal 已提交，`configs/` 里的默认参数和 proposal 一致。

---

## 3. 10/31–11/6：harness 完整版 + baseline + 开始收日志

**步骤**

1. **检索器**（10/31）：`retrieval/retriever.py`，BM25（rank_bm25）+ embedding（all-MiniLM-L6-v2）+ RRF，返回固定长度 3，不足用空位补齐；持久模式排除已在上下文中的记录。embedding 在生成世界时算好存盘。
2. **状态与特征**（11/1–11/2）：`env/state.py`、`advance()`、`env/features.py`（50 维观测，标出 H3 消融要置零的历史特征）。
3. **环境**（11/2–11/3）：`env/gate_env.py`（`Discrete(8)` 动作）+ `env/backend_live.py`（同步包装异步客户端）。
4. **全部单元测试**（11/3）：生成器、打分器、状态更新、环境观测范围、`mask_history`、空位动作。
5. **runner**（11/3）：`harness/runner.py`，session 间并发，失败的 session 整个丢弃，每 100 个 session flush 一次 parquet。
6. **baseline**（11/4）：`policies/heuristics.py`（always / never / threshold）、`policies/linucb.py`（含风险敏感版本）。
7. **开始收日志**（11/4 起，后台跑）：`collect_logs.py`，只用训练用户，70% 均匀随机 + 30% 阈值策略，持久和一次性模式各 1,500 session。
   - 先各跑 100 个，检查日志字段齐全、成本符合预算，再放开跑。
8. **人工标注打分器**（11/5–11/6）：从 E0 和日志里抽 200 条回答人工标注。

**关卡 G2（11/6）**

- `pytest` 全绿。
- 打分器一致率：答对与旧值判定 ≥ 95%，无关提及 ≥ 90%（不够就补别名表或收紧回答格式，再标一次）。
- 日志收集已在后台稳定跑，按当前速度能在 11/9 前完成。

---

## 4. 11/7–11/13：模拟器 + DP（第二个关键节点）

**步骤**

1. **行为模型**（11/7–11/8）：`sim/outcome_features.py`、`sim/behavior_model.py`，三个逻辑回归子模型（回答结果三分类、提及二分类、回答长度回归）。
2. **模拟器验证**（11/9）
   - 留出 20% session：对数似然、校准曲线。
   - 同一行为策略在模拟器上重跑，回报各分项分布和真实日志对比。
   - 对照梯度提升树版本；明显更准就补交互特征。
3. **sim 后端**（11/10）：`env/backend_sim.py`，和 live 后端共用打分与状态更新代码；写测试保证两者对同一组结果输出完全相同。
4. **小实例 DP 与贪心**（11/10–11/11）：`planning/abstract.py`、`dp.py`、`greedy.py`。
5. **必须成立的关系**（11/12），全部写成 pytest：
   - V^DP ≥ V^greedy，且 ≥ 任何启发式的精确价值；
   - T = 1 时 V^DP = V^greedy；
   - λ = α = δ = 0 且提及概率与历史无关时 Δ = 0；
   - 小实例上 DQN γ=0 接近贪心、γ=1 接近 DP；
   - 采样估计价值与递归精确值的差在置信区间内。
6. **第一版 γ 扫描**（11/13）：`train.py`，DQN × γ ∈ {0, 0.3, 0.6, 0.9, 1.0} × 1 个种子，在模拟器上跑。

**关卡 G3（11/13）**：上面 5 条关系全部成立；第一版 γ 扫描出图。任何一条不成立都先查 bug，不做主实验。

---

## 5. 11/14–11/20：主实验

1. 只在 γ = 0.9 上对学习率和网络宽度做一次小网格搜索，其他 γ 沿用。
2. **E1 + E2**：全部方法、全部 γ、5 个种子（`e1_main.yaml`、`e2_gamma.yaml`）。
3. **E3**：`solve_dp.py`，T ∈ {4, 6, 8, 10} × 过时比例 {0, 0.2, 0.4, 0.6}，两种模式；完整规模用 DQN γ=0.9 与 γ=0 的差近似（4–20 轮）。
4. **E4 消融**：`mask_history=True` 重训。
5. **E5**：一次性模式下重跑 E1、E2、E3。
6. `evaluate.py --backend sim --split test --n 2000`，汇总成 `results/sim_eval.parquet`。

**关卡 G4（11/20）**：表 1、图 2、图 3 都能从结果文件生成；每个数字带 5 种子的均值和 bootstrap 95% 置信区间。

---

## 6. 11/21–11/27：真实 LLM 验证

1. **E6**：每种方法取验证集最好的种子，在 DeepSeek 上各跑 300 个测试 session（`evaluate.py --backend live`）。开跑前先确认 `model` 字段和日志收集时一致。
2. 计算模拟器与真实 LLM 的方法排名 Spearman 相关。
3. α、λ 敏感性分析（模拟器上即可）。
4. 可选 **E7**：换一个更强的模型重复 E0 和 E6。

**关卡 G5（11/27）**：`results/live_eval.parquet` 完整；排序一致性已算出；不一致的地方记下来写进局限性。

---

## 7. 11/28–12/7：报告与展示

1. `make_figures.py` 一键生成图 1–3、表 1、案例图。
2. 报告初稿：正文 5 页 + 附录（模拟器验证、sim vs real、敏感性分析、DeepSeek 版本说明）。
3. 10 分钟展示 slides，展示前提交 Gradescope。

**关卡 G6（12/7）**：从干净的 clone 出发，按 README 能复现全部图表（真实 LLM 部分走缓存）。

---

## 8. 12/10 17:00：提交最终报告

---

## 每天的固定动作

- 提交前跑 `pytest` 和密钥检查（第 0 节第 4 条）。
- 每个实验的配置副本随结果存进 `runs/<实验>/`。
- 记录当天的 DeepSeek token 用量和花费，超出预算 20% 就停下来重新估算。

## 风险速查

| 风险 | 应对 |
| --- | --- |
| E0 不过关 | 见 G0 的处理顺序 |
| DeepSeek 限流或高峰期变慢 | 降并发；日志收集放夜间跑；SQLite 缓存保证中断后不重复付费 |
| `deepseek-chat` 底层版本变化 | 每次调用记录 `model` 字段；关键阶段集中跑；写进局限性 |
| 时间花在搭环境上 | harness 控制在 300–500 行；11/13 前必须出第一版 γ 扫描 |
| key 泄露 | `.gitignore` + 提交前 grep；一旦误提交，立即去 DeepSeek 后台作废并换新 key，再清理历史 |
