# memgate 执行计划（一次写完版）

这份文档是动手写代码之前的**完整上下文**：所有需要拍板的决定先定死，所有模块的接口和数据格式先写清，所有能预见到的问题先列出来并给出处理方式。写代码时照着它一口气写完整个工程，不做"先跑通一个最小版再说"。

依据：[docs/项目文档.md](docs/项目文档.md)（做什么、为什么、假设与实验设计）和 [docs/工程实现文档.md](docs/工程实现文档.md)（模块代码骨架、超参表、成本估算）。本文是在它们之上的增量：与原文档的差异全部集中在 §1.2，补充的规则在 §1.4–§1.6、§2、§3.5a 等处，其余以原文档为准；两者冲突时以本文为准。写代码时三份一起读。

**核心论点**：注入的 memory 会在一定程度上改变模型之后的输出，比如把回答锚定在注入的内容上、限制模型的回答、让模型反复提起同一件事。这些影响会跨轮持续，所以门控需要用 RL 做长期规划，而不是逐轮当 bandit 处理。

为了让这个论点站得住，结论必须证明长期规划的价值**来自 LLM 自身的行为**，而不只是来自我们设计的奖励（按轮计费的 token 成本、逐次加重的打扰惩罚），也不能被一个"把成本按剩余轮数摊销"的简单修正找回来。对应的设计见 §1.6 的奖励变体、§3.11 的摊销贪心、§3.13 的主结论对比。

目录：

0. 核心论点（上段）
1. 定死的决定
2. 跨模块契约（数据结构、编码、日志、配置）
3. 模块规格（每个模块：做什么、接口、实现要点）
4. 问题清单（按类别，共 80 余条，每条带处理方式）
5. 测试清单（写代码时同步写完）
6. 写代码的顺序与运行顺序
7. 时间线与关卡
8. 写代码之前还要确认的事

---

## 1. 定死的决定

### 1.1 环境与依赖

| 项目 | 决定 |
| --- | --- |
| Python | 3.11（Mac 上用 `brew install python@3.11` 或 pyenv） |
| 包管理 | `pyproject.toml` + `pip install -e ".[dev]"`，虚拟环境放 `.venv/` |
| 核心依赖 | `openai>=1.40`、`python-dotenv`、`pydantic>=2`、`pyyaml`、`numpy`、`pandas`、`pyarrow`、`scikit-learn`、`gymnasium>=0.29,<1.1`、`stable-baselines3>=2.3`、`torch`、`sentence-transformers`、`rank-bm25`、`tiktoken`、`matplotlib`、`scipy`、`tqdm` |
| 开发依赖 | `pytest`、`pytest-asyncio`、`ruff` |
| 版本锁定 | 装好后 `pip freeze > requirements.lock`，提交进 git |
| 随机数 | 全部用 `np.random.default_rng(seed)` 显式传递；不用 `random` 模块和 `np.random.seed` 全局状态 |
| 哈希 | 一律 `hashlib.sha256`，**绝不用 Python 内置 `hash()`**（每次进程启动随机化，会让缓存和 ID 不可复现） |

### 1.2 与原文档不同的地方

| 编号 | 原文档 | 本文决定 | 原因 |
| --- | --- | --- | --- |
| D1 | 自部署 Qwen3-8B 或托管 API | DeepSeek API，`deepseek-chat`（非思考模式） | 用户决定 |
| D2 | 一次性模式把 memory 加在 system prompt 末尾 | 一次性模式把 memory 拼在**当前 user 消息前面**（和持久模式同一个位置），下一轮从历史里删掉 | ① 放在 system prompt 会让每轮的第一条消息都变，DeepSeek 的前缀缓存完全失效，成本翻几倍；② 两种模式只差"是否留在历史里"这一个变量，H4 的对照更干净 |
| D3 | 调用时传 seed | 不依赖 seed；复现靠本地 SQLite 响应缓存 | DeepSeek 不保证 seed 生效 |
| D4 | 并发信号量 32 | 默认 8，E0 时按实测调 | 托管 API 的限流与排队行为未知 |
| D5 | token 数没指定怎么算 | 用 `tiktoken` 的 `cl100k_base` 统一计数（只作代理，用于奖励里的 λ 项）；API 实际计费用响应里的 usage | 本地需要一个确定、快速的计数器；DeepSeek 的 tokenizer 不方便离线用 |
| D6 | 未定义用户数和 session 数 | 见 §1.4 | 需要定死 |
| D7 | 未定义"过时比例"的具体含义 | 见 §1.5 | E3 扫描需要可操作的定义 |
| D8 | E0 用默认随机 session | E0 用专门设计的固定结构 session，见 §3.9 | 随机 session 测不准"注入后第 k 轮"和"纠正后" |

### 1.3 LLM 调用参数（configs/llm.yaml）

```yaml
base_url: https://api.deepseek.com
model: deepseek-chat
api_key_env: DEEPSEEK_API_KEY       # 只写变量名，不写 key
temperature: 0.7                    # 见问题 A6：DeepSeek 的温度语义要在 E0 实测
top_p: 0.95
max_tokens: 160
timeout_s: 120
max_retries: 5
concurrency: 8
system_prompt_version: v1           # v1 = 带"只在相关时使用"；v2 = 去掉这句（G0 不过时用）
cache_path: cache/llm_cache.sqlite
```

### 1.4 规模（configs/world.yaml）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 用户数 | 100 | 训练 / 验证 / 测试 = 60 / 20 / 20 |
| 每用户 key | 14：会变 6、偏好 4、琐事 4 | 同原文档 |
| 会变事实的历史版本数 | 由过时比例 p_stale 控制，见 §1.5 | |
| session 长度 T | 默认 12；E3 扫 4–20 | |
| 轮类型概率 | 提问 0.40 / 闲聊 0.45 / 更新 0.15 | 同原文档 |
| 闲聊中带同主题词的比例 | 0.5 | 制造"看起来相关、实际无关"的候选 |
| 固定脚本集 | 每个验证用户 50 个、每个测试用户 100 个 session 脚本 | 测试共 2,000 个，所有方法在同一组上评估 |
| 训练脚本 | 不预先固定：模拟器训练时每次 `reset` 从训练用户现场生成一个新脚本 | 无限多样，避免过拟合到固定脚本 |
| 日志收集脚本 | 从训练用户预先生成 3,000 个，固定下来 | 两种模式各 1,500 |
| 世界 seed | 0 | 再跑一个 seed=1 的世界作稳健性检查（可选） |

### 1.5 过时比例的定义

p_stale = **会变事实在 session 开始时已有至少一个历史版本的比例**。生成用户时，对每个会变 key 以概率 p_stale 生成 1–3 个历史版本（均匀抽版本数），否则只有当前版本。session 内的更新轮还会额外制造过时版本，这部分不受 p_stale 控制。默认 p_stale = 0.5；E3 扫 {0, 0.2, 0.4, 0.6}。

小实例 DP 里同样定义：A、B 两个 key 各以 p_stale 的概率在开始时带一个过时版本，DP 对这 4 种初始情况按概率加权求期望。

### 1.6 奖励（configs/reward.yaml）

δ = 1.0，α = 0.3，β = 1.5，λ = 0.002 / token（同原文档）。补充定死：

- 只有提问轮有 r_quality 和 r_stale；闲聊和更新轮这两项为 0。
- n_m 用**本轮打分之前**的提及次数，打分之后再加 1。
- token 项按"本轮结束时在上下文中的全部 memory"计：持久模式 = 已在上下文 ∪ 本轮注入；一次性模式 = 本轮注入。
- 折扣：报告里的 session 回报一律是**不折扣的总和**（γ 只是训练超参），否则不同 γ 的回报不可比。

**奖励变体**（`reward.yaml` 里三个命名配置，DP、训练、评估都可以指定）：

| 名字 | 系数 | 用途 |
| --- | --- | --- |
| `R_full` | δ=1.0, α=0.3, β=1.5, λ=0.002 | 默认，全部主实验 |
| `R_notok` | 同上，但 λ=0 | 去掉 token 成本这一条人为的跨轮结构 |
| `R_behavior` | 同上，但 λ=0、β=1 | 再去掉"逐次加重的打扰惩罚"。奖励本身没有任何跨轮结构，剩下的长期影响只来自 LLM 的行为（锚定旧值、重复提及、复用） |

注意：换奖励变体只改打分，不改 LLM 行为，所以**不需要重新收日志**，行为模型也不用重新拟合。

---

## 2. 跨模块契约

### 2.1 ID 和值

- 用户 ID：`u000`–`u099`；记录 ID：`u003-m017`；session ID：`{split}-{user}-{序号:04d}`，例如 `test-u083-0042`。
- 每个 value 在**整个世界**里唯一，且满足：规范化后不是任何其他 value 或别名的子串（按词边界）；不是常见英文词。
- 别名表 `aliases: dict[value, list[str]]`，生成时一起存盘。

### 2.2 动作编码

动作 a ∈ {0..7}，第 i 位（`(a >> i) & 1`）表示注入第 i 个候选（i = 0 是融合排名第 1）。空位对应的位忽略。所有模块只用 `decode_action(a, cands) -> list[Candidate]` 这一个函数解码。

### 2.3 观测向量布局（50 维，float32，全部裁剪到 [−1, 1]）

| 下标 | 内容 | 归一化 | 历史 |
| --- | --- | --- | --- |
| 候选 i 的偏移 = 14·i，i = 0,1,2 | | | |
| +0 | 位有效 | 0/1 | |
| +1 | 余弦相似度 | 原值（[−1, 1]） | |
| +2 | BM25 分 | 本轮候选里除以最大值 | |
| +3 | 融合排名 | (rank−1)/2 | |
| +4 | 记录年龄 | min(天数/365, 1) | |
| +5 | token 数 | min(tokens/60, 1) | |
| +6..+10 | 类别 one-hot | 0/1 | |
| +11 | 候选中有同 key 更新的版本 | 0/1 | |
| +12 | 同 key 的其他版本已在上下文 | 0/1 | 是 |
| +13 | 本 session 此前作为候选出现次数 | min(n/5, 1) | 是 |
| 42 | t/T | | |
| 43 | (T−t)/T | | |
| 44 | 上下文 memory 条数 | min(n/10, 1) | 是 |
| 45 | 上下文 memory token 总数 | min(n/300, 1) | 是 |
| 46 | 上一个回答里提到的上下文 memory 数 | min(n/3, 1) | 是 |
| 47 | 单条最大被提及次数 | min(n/5, 1) | 是 |
| 48 | 上下文中有更新版本出现过的 memory 条数 | min(n/3, 1) | 是 |
| 49 | 上一轮是否有更新与上下文冲突 | 0/1 | 是 |

`HISTORY_IDX` 常量列出全部"历史"下标，`mask_history=True` 时这些位置置 0。

注意 48：控制器看不到 `superseded_by`，但能看到"store 里出现过同 key、时间更晚的记录"，这是 harness 部署时也能拿到的信息。实现时只用 `created_at` 和 key，**不读 `superseded_by`**。

### 2.4 结果对象

```python
@dataclass(frozen=True)
class Outcome:
    answer: Literal["correct", "stale", "neither", "na"]   # na = 非提问轮
    mentioned_ids: frozenset[str]        # 回答里出现了 value 的全部 memory（含相关的）
    intrusion_ids: frozenset[str]        # 其中算作无关提及的
    r_quality: float; r_stale: float; r_intrusion: float; r_tokens: float
    @property
    def reward(self) -> float: ...
```

live 后端：LLM 回答 → 打分器 → Outcome。sim 后端：行为模型采样 answer 和 mentioned_ids → **同一个打分器的奖励函数** → Outcome。两者共用 `compute_reward(st, turn, chosen, answer, mentioned_ids, mode, cfg)`。

### 2.5 日志（每轮一行 parquet）

在原文档 TurnLog 的字段之外，**额外存一份状态快照**，这样之后改行为模型特征时不用重新调 LLM：

| 新增字段 | 内容 |
| --- | --- |
| `state_json` | 本轮决策前的 in_context、mentions、本 session 用户说过的值、上一轮回答提到的 id、conflict_flag |
| `candidate_ids`、`candidate_raw` | 候选 id 和原始检索分数（未归一化） |
| `llm_model`、`system_fingerprint` | 响应里返回的模型标识 |
| `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens` | DeepSeek usage 里的缓存字段（缺失时记为空） |
| `finish_reason` | `stop` / `length` / 其他 |
| `latency_s`、`created_at_utc` | 用来排查限流和版本变化 |
| `world_seed`、`config_hash`、`git_commit` | 复现用 |
| `prompt_version`、`behavior_policy` | `uniform` / `threshold` / 方法名 |

### 2.6 配置

每个实验一个 YAML，用 pydantic 模型校验（未知字段报错）。运行时把合并后的完整配置、`git rev-parse HEAD`、`pip freeze` 写进结果目录。

---

## 3. 模块规格

### 3.1 world/pools.py、templates.py

- 词表全部虚构：项目代号（鸟名 + 罕见词，如 "Kestrel"、"Quillon"）、会议室（"Halcyon Room"）、人名（罕见组合，如 "Ottoline Varga"）、日期、工具名（虚构，如 "Brindle Notes"）、宠物名、咖啡馆名。
- 每类准备的值数量 ≥ 用户数 × 每用户需要的数量 × 3（更新要抽未用过的值）。
- **偏好也用专名值**（例如偏好的写作风格是 "the Marlowe style"，常用工具是 "Brindle Notes"），不要用"简洁的风格"这种模型会意译的描述，否则字符串匹配判不出来。
- 日期别名统一生成：`Friday 14 November`、`14 November`、`November 14`、`Nov 14`、`14 Nov`、`November 14th`、`14th November`、`Nov. 14`。星期几单独出现不算命中。年份统一不写。
- 同一月份内的两个日期别名会互相包含吗？"November 1" 是 "November 14" 的前缀，所以**匹配必须按词边界**（见问题 C2），并且生成器检查规范化后的别名两两不按词边界包含。
- 模板：提问 ≥ 8 种句式、闲聊 ≥ 20 种（一半带主题词槽位）、更新 ≥ 6 种。闲聊文本里**不能出现任何 value**。

### 3.2 world/generator.py

- 输入：`WorldConfig`、seed。输出：`data/world_s{seed}/` 下 `users.json`、`records.parquet`、`aliases.json`、`scripts_{split}.jsonl`、`embeddings.npy`（records 和全部固定脚本的轮文本）、`turn_index.json`。
- 记录的 `text` 用固定句式："{subject}'s {attribute} is {value}."，同 key 的新旧版本只差 value。
- `tokens` 字段在生成时用 tiktoken 算好。
- 训练用户的脚本生成函数要能在环境 `reset` 时现场调用（不读盘），并且同一 (seed, 计数器) 产出相同脚本。现场生成的轮文本没有预计算 embedding，见问题 D3。
- 生成后立即运行 `validate_world()`：别名不重叠、闲聊不含 value、每个提问轮的 answer_value 等于该时刻 store 里该 key 最新的记录、每个 split 的用户不重叠。不通过就抛异常。

### 3.3 retrieval/retriever.py

- `topk(query_text, query_emb, store, exclude) -> list[Candidate | None]`，长度固定 3。
- BM25 用 `BM25Plus`（不是 `BM25Okapi`，见问题 D1），语料 = 该用户当前 store 的 text，分词 = 规范化后按空格切。
- RRF：`score = 1/(60 + rank_bm25) + 1/(60 + rank_emb)`；平分时按记录 id 的 sha256 打破（确定、但不偏向新或旧）。
- 不过滤过时记录；持久模式排除 `exclude`。
- 性能：store 变化只发生在更新轮，按 (user, store 版本) 缓存 BM25 索引；embedding 相似度是一次 (1×384)·(384×n) 乘法。

### 3.4 harness/llm_client.py

- `AsyncOpenAI(base_url, api_key=os.environ[...], timeout=timeout_s, max_retries=0)`：SDK 自带重试关掉，统一用自己的重试（指数退避 1, 2, 4, 8, 16 秒 + 0–1 秒抖动，最多 5 次）。
- 可重试：超时、连接错误、429、500、502、503；不可重试：400、401、402（余额不足）、422，直接抛出并让整个 runner 停下（余额不足时继续跑只会丢一堆 session）。
- 缓存 key = sha256(model, messages 的规范 JSON, temperature, top_p, max_tokens, session_id, turn_idx, sample_idx)。**必须包含 session_id 和轮次**，否则两个 session 走到完全相同的消息时会拿到同一个缓存回答，破坏随机性。
- 返回 `LLMResponse(text, model, system_fingerprint, prompt_tokens, completion_tokens, cache_hit, cache_miss, finish_reason, latency_s)`。
- 记录每次调用的 usage 到一个累计计数器；超过 `budget_usd` 配置就停下。

### 3.5 harness/prompt.py

```python
SYSTEM_V1 = ("You are a helpful personal assistant. Notes about the user may appear inside "
             "<memory> tags. Use a note only if it is relevant to the user's current request. "
             "Answer in at most 80 words.")
SYSTEM_V2 = ("You are a helpful personal assistant. Notes about the user may appear inside "
             "<memory> tags. Answer in at most 80 words.")

def build_messages(history, turn, chosen, mode) -> list[dict]:
    # history: 已经发生的 (user_content_as_stored, assistant_text) 对
    # 本轮 user 内容 = "<memory>\n- ...\n</memory>\n\n" + turn.text（chosen 为空时不加标签）
    # 持久模式：本轮 user 内容原样存进 history
    # 一次性模式：存进 history 的是去掉 <memory> 块的 turn.text
```

注入块里的条目顺序固定为融合排名顺序。

### 3.5a memory 与 harness 的边界规则

原文档对 harness 的主流程写得很清楚，但下面这些边界情况没有明说，写代码时会碰到，这里一次定死：

| # | 规则 | 理由 |
| --- | --- | --- |
| M1 | **写入是脚本化的**：更新轮的新记录由生成器给出（`turn.new_record`），harness 不用 LLM 从对话里抽取事实 | 只研究"放什么"，把"写错"排除在外；报告里作为局限性说明 |
| M2 | 旧记录**不删除、不标记**，仍留在 store 里，照常可被检索；`superseded_by` 只给打分器和 DP 用 | 这就是过时信息进入候选的途径，和大多数真实 harness 一致 |
| M3 | 注入文本是 `- {record.text}`，**不带日期、版本号或"已更新"字样** | 带日期会让模型自己判断新旧，削弱过时效应；带日期的版本只作为 J1 之外的一个可选敏感性实验 |
| M4 | 控制器**每一轮都做决策**，包括闲聊轮和更新轮 | 更新轮注入旧版本会直接造成冲突，这正是要学的 |
| M5 | 检索的查询只用**当前这一轮的用户输入**，不拼历史 | 和常见 harness 一致；也让检索结果只依赖脚本，可以预计算 |
| M6 | 持久模式下注入的 memory **不能撤回**，没有"删除"动作 | 这是"本轮不可撤销"的假设本身；H4 用一次性模式作对照 |
| M7 | 一次性模式每轮从**全部 store** 检索，同一条记录可以连续多轮被注入，每次都计 token | 和原文档一致 |
| M8 | 同一个 key 的新旧版本**可以同时注入**（两条都在候选里时） | 不替控制器做去重，冲突要靠它自己学会避免 |
| M9 | 每个 session 都从该用户的**初始 store** 开始；session 内更新写入的记录只在本 session 内有效，不带到下一个 session | session 之间独立，episode 定义才干净 |
| M10 | 用户的更新消息在两种模式下都**留在历史里**，所以模型可以从对话本身知道新值 | 真实情况就是"memory 说旧值、对话里说过新值"的冲突；行为模型里有"用户是否在本 session 说过新值"这个特征 |
| M11 | 模型只生成文字回答：没有工具调用，也不能自己写 memory | harness 保持在 300–500 行 |
| M12 | token 成本只算 memory 条目本身的 token，`<memory>` 标签和换行忽略 | 每轮常数，不影响决策 |

### 3.6 harness/scorer.py

- `normalize(s)`：小写 → 把 `’` 换成 `'` → 去掉除字母数字和空格以外的字符 → 合并空白 → 两端各补一个空格。
- `hit(v, resp_norm)`：v 的任一规范化别名 `" " + a + " "` 是 resp_norm 的子串（词边界匹配）。
- `classify_answer`：提问轮里，当前值命中 → correct；否则任一过时值命中 → stale；否则 neither。
- `mentioned_ids`：store 里 value 命中的全部记录。`intrusion_ids` = mentioned 中 id ∉ relevant_ids 且 key ≠ target_key，且**该 value 没有出现在本轮用户输入里**（用户自己说的值被复述不算打扰）。
- 另外记录一个诊断标志 `hedged`：当前值和过时值同时命中（不进奖励，只用来分析"两个都说"的情况，见问题 C5）。

### 3.7 env/state.py、features.py、gate_env.py

- `SessionState` 在 sim 训练里**原地更新**（`advance_` 带下划线），只有 `fork()` 才 deepcopy。原文档里 `st = advance(st, ...)` 若每步都复制，30 万步会明显变慢。
- `advance_` 依次：记录本轮候选出现次数 → 更新 mentions（用 Outcome.mentioned_ids）→ 持久模式把 chosen 加进 in_context / 一次性模式清空后只放本轮（仅用于本轮计费）→ 更新轮把 new_record 写进 store → 更新 conflict_flag → t += 1。
- `terminated = (t == T)`，`truncated = False`。时间在状态里，所以这是真正的终止，不是截断。
- `observation_space = Box(-1, 1, (50,), float32)`；`_obs()` 最后 `np.clip`，并 assert 无 NaN。
- live 后端在环境里只用于小规模调试；大批量真实调用一律走 runner（并发）。

### 3.8 harness/runner.py

- `run_sessions(scripts, policy_factory, mode, ...)`：每个 session 内顺序执行，session 之间 `asyncio.gather` + 信号量。
- 每个 session 结束后才把它的全部 TurnLog 放进写缓冲；失败的 session 丢弃并写入 `failed.jsonl`（session_id、错误类型、轮次）。
- 每 100 个 session 写一个 parquet 分片 `part-00000.parquet`，**不追加写同一个文件**（parquet 不支持追加；中途崩溃也只损失最后一片）。
- 启动时读取已有分片里的 session_id，跳过已完成的（断点续跑）。
- 进度条显示：完成 session 数、失败数、累计 token、累计费用估计、缓存命中率。

### 3.9 scripts/run_pilot.py（E0）

E0 用固定结构的 8 轮 session，每个 session 只针对一个目标 key K 和一条干扰记录 R：

| 轮 | 类型 | 说明 |
| --- | --- | --- |
| 0 | 提问 K | **唯一一次注入**，注入内容由条件决定（四种条件下第 0 轮都是同一句提问） |
| 1 | 闲聊（与 R 同主题） | 测 R 的重复提及 |
| 2 | 闲聊（无关） | |
| 3 | 更新 K（用户说出新值） | 过时条件下就是"纠正" |
| 4 | 闲聊（无关） | |
| 5 | 提问 K | 测纠正后旧值使用 |
| 6 | 闲聊（与 R 同主题） | |
| 7 | 提问 K | |

四种条件：不注入 / 注入 K 的当前值 / 注入 K 的过时值 / 注入无关记录 R。第 1–7 轮都不注入。每条件 50 个 session，持久模式。另外用一次性模式各跑 20 个作参照（H4 的早期信号，便宜）。

**对照组怎么选（分析脚本里写死，并在 summary 里注明）**：

- 第 3 轮更新之后，第 0 轮注入的"当前值"也变成了过时值。所以在第 5、7 轮，"注入当前值"和"注入过时值"两个条件的上下文里**都有旧值**，"注入当前值"不是干净的对照。
- "纠正后旧值使用率"（第 5、7 轮）的对照组一律用**"不注入"**条件；比较对象是"注入过时值"和"注入当前值"，两者都相对"不注入"报差值。
- "注入当前值"在第 5、7 轮单独作为一种情况报告：**注入时正确、后来过时**。它和"注入时就过时"的差别本身有意思（模型在第 0 轮用过这个值、并且当时是对的），但不能当对照。
- "R 的重复提及率"（第 1、2、4、6 轮）的对照组同样是"不注入"（不注入时 R 也可能因为闲聊同主题被模型编出来或碰巧提到，这是基线）。
- 第 0 轮本身（注入当轮）的答对率和旧值率单独报告，不和后续轮混在一起。

输出：`runs/e0/turns.parquet`、`runs/e0/summary.csv`（每条件 × 每轮：R 被提及率、旧值使用率、答对率，带 bootstrap 置信区间，以及相对"不注入"的差值）、`runs/e0/fig1.png`。

继续条件（G0）照原文档。

### 3.10 sim/behavior_model.py

- 特征函数 `outcome_features(state_snapshot, turn, chosen, mode)` 只接受**快照**（§2.5 的 state_json 能还原的东西），日志和模拟器共用。
- 三分类回答模型、提及模型（每条"候选可提及"memory 一个样本：在上下文中的 + 之前被模型提到过的 + 本轮候选中被注入的）、长度回归。
- 提及模型的样本里**排除本轮 relevant 的记录**（那些被提到是正常回答，不是打扰），只学无关提及。
- 标准化：`StandardScaler` + `LogisticRegression(C 在验证集上选)`，存 `runs/sim/model.pkl`（连同特征名列表和 sklearn 版本）。
- `probs(...)` 返回解析概率给 DP；`sample(..., rng)` 给 sim 后端。

### 3.11 planning/

- `abstract.py`：小实例状态 `AbsState(t, in_ctx: 5 位, mentions: 每条 0/1/2, updated_A, updated_B, init_stale: 2 位)`，用 NamedTuple 保证可哈希。
- `abstract_to_features(...)`：把抽象状态映射成行为模型需要的特征向量。小实例里的相似度、BM25、类别等用固定常数（取日志中对应请求类型的中位数），在 `abstract.py` 里写死成表。
- `dp.py`：`lru_cache` 递归；T ≤ 10 不会碰到递归深度问题。结果数 = 3 × 2^5 = 96，按"每条记录独立提及"展开。
- 策略评估：`V_pi(s)` 用同一个递归，只是 max 换成按策略取动作。DP 最优、贪心、摊销贪心、各启发式都这样精确算。
- **分项价值**：策略评估时把四个奖励分项各自累加，得到 V_π = V_π^quality + V_π^stale + V_π^intrusion + V_π^tokens，所以 Δ 可以按分项拆开。
- `greedy.py` 里有两个短视策略：
  - **贪心**：只看本轮期望奖励。
  - **摊销贪心**：和贪心一样只看一步，但把本轮新注入的 memory 的 token 成本乘以剩余轮数，即 λ·tokens(m)·(T−t)（持久模式；一次性模式下它和贪心相同）。这就是"知道 memory 会一直占地方"的短视策略。它能修正 token 成本带来的长期结构，但修正不了 LLM 行为带来的长期影响（锚定、重复提及、复用的时机）。
- 完整规模上也实现摊销贪心：用行为模型的 `probs` 算本轮期望奖励，再加摊销后的 token 成本，作为一个不需要训练的策略参与评估。

### 3.12 policies/

- 统一接口 `Policy.act(obs: np.ndarray, info: dict) -> int`。`info` 里带候选元数据，只给 threshold 用（它要融合相似度原值）。
- threshold：注入融合分 ≥ τ 的候选。τ 在验证用户上网格搜索（20 个分位点），每种模式单独调。
- LinUCB：8 个动作各一个 `A = I·reg + Σxxᵀ`、`b`；用 Sherman–Morrison 更新逆矩阵；训练在模拟器上做（和 DQN 同样的 30 万步），评估时冻结。
- sb3.py：DQN 和 PPO 的构造、训练、加载；每个 run 设 `torch.set_num_threads(1)`，用多进程并行跑多个 run。

### 3.13 analysis/

- `stats.py`：按 session bootstrap（1,000 次）的均值置信区间；方法间用同一组 session 的配对差值；5 个种子先在种子内平均，再跨种子报告均值 ± 标准差。
- `plots.py`：图 1–3、表 1、案例图，只读 `runs/` 和 `results/`，不重新计算任何东西。
- **哪些对比可以支撑主结论**（写进 `stats.py` 的常量 `PRIMARY_CONTRASTS`，报告正文只引用这些）：
  1. Δ = V^DP − V^greedy（小实例，精确值），纯短视损失，并且**按四个奖励分项拆开**；
  2. **Δ_amort = V^DP − V^amortized-greedy**：把 token 成本按剩余轮数摊销之后还剩下的短视损失。它支撑"需要 RL 做长期规划"这个核心论点；如果它接近 0，结论就是"摊销一下成本就够了"；
  3. **行为驱动的 Δ**：在奖励变体 R_behavior（λ = 0、β = 1，见 §1.6）下重新求解得到的 Δ。这时奖励里已经没有任何跨轮结构，剩下的差距只能来自 LLM 自身的行为；
  4. DQN γ=0.9 − DQN γ=0（完整规模，配对差值），在完整奖励和 R_behavior 下各报一次；
  5. 上面各项在持久模式与一次性模式之间的差（H4）；
  6. DQN γ=0.9 − DQN γ=0.9 去掉历史特征（H3）。
- 报告正文的主张按强弱分三档：Δ_amort 和行为驱动的 Δ 都显著大于 0 → "LLM 行为本身需要长期规划"（最强）；只有 Δ 显著、Δ_amort 接近 0 → "长期结构主要来自成本，摊销即可"；Δ 也接近 0 → "bandit 简化合理"。
- LinUCB、风险敏感 bandit、阈值、always/never 是**参照**：说明已有方法和常见做法落在哪里，不用来证明长期规划有用，因为它们和 DQN 之间除了 γ 还差探索方式和模型容量。

---

## 4. 问题清单

按类别列出，每条：可能出什么问题 → 怎么处理。写代码时逐条对照。

### A. DeepSeek API

| # | 问题 | 处理 |
| --- | --- | --- |
| A1 | `deepseek-chat` 是别名，官方升级时底层模型会变，前后两批数据不可比 | 每次调用记录 `model` 和 `system_fingerprint`；日志收集和 E6 之前各检查一次；变了就在报告局限性里写明，必要时重收日志 |
| A2 | 误用 `deepseek-reasoner`：输出带思考过程、慢、贵、打分被思考文本干扰 | 配置里写死 `deepseek-chat`；client 启动时断言模型名不含 `reasoner` |
| A3 | seed 不生效，同样的请求结果不同 | 不依赖 seed（D3）；复现靠缓存 |
| A4 | 高峰期排队，请求挂很久才返回或断开 | `timeout_s=120`；超时可重试；runner 记录 latency，持续变慢就降并发；大批量放在国内夜间跑 |
| A5 | 429 / 503 | 指数退避重试；连续失败率 > 5% 时 runner 自动把并发减半 |
| A6 | DeepSeek 的 temperature 语义可能和 OpenAI 不完全一样（官方文档曾说明 API 温度会被映射） | 开工前查官方文档当前说明；E0 里看同一条件下的回答方差，如果几乎确定性就调高温度，因为模拟器要学的是概率 |
| A7 | 余额不足（402）时继续跑会丢掉大量 session | 402 视为致命错误，立即停下 runner |
| A8 | 回答为空、`finish_reason="length"` 被截断、或触发内容过滤 | 空回答重试一次，仍为空就记为 neither 并标记；截断照常打分（80 词要求通常不会超 160 token），统计截断率，> 2% 时把 max_tokens 调到 200 |
| A9 | 前缀缓存不命中，成本远超预算 | D2 保证前缀稳定；日志记 `prompt_cache_hit_tokens`；E0 后算实际命中率，低于 50% 先排查消息拼装再继续 |
| A10 | 一次性模式下"删掉历史里的 memory 块"本身会让前缀变化 | 因为删的是上一轮 user 消息里的块，从那一条起的前缀会变：每轮只有最近一对消息不能命中，影响很小；接受 |
| A11 | 价格和优惠时段会变 | 开工前查价格页，写进 `llm.yaml` 注释；`budget_usd` 设上限 |
| A12 | key 泄露 | 只从环境变量读；`.env` 在 `.gitignore`；缓存和日志里不写请求头；提交前 grep `sk-` |
| A13 | 回答用中文或 Markdown（加粗、列表） | system prompt 和对话全英文；打分前 normalize 会去掉 Markdown 符号；统计非英文回答比例 |
| A14 | 模型在提问轮反问（"Which project do you mean?"）而不是回答 | 记为 neither；问句模板里总是写明实体名，减少歧义 |

### B. 世界生成

| # | 问题 | 处理 |
| --- | --- | --- |
| B1 | 虚构名字恰好是真实存在的东西，模型用自己的知识回答 | 名字用罕见组合；E0 的"不注入"条件里答对率应接近 0，若不是就查哪些值被猜中 |
| B2 | 值被模型意译，字符串匹配判错 | 所有值都是专名（§3.1）；人工标注 200 条时专门统计这类错误 |
| B3 | value 之间按词边界互相包含（"Nov 1" 与 "Nov 14"、"Oak" 与 "Oak Hall"） | 生成时检查；日期别名不含"Nov 1"这种短形式之外的歧义；不通过就重抽 |
| B4 | 闲聊模板里不小心带了 value，导致"用户自己说的"被当成打扰 | `validate_world()` 检查；打分器也排除用户本轮说过的值 |
| B5 | Zipf 抽样让少数 key 占了几乎所有提问，其他 key 的数据不够 | Zipf 指数取 1.0，并检查每个 key 在日志里至少出现若干次 |
| B6 | 更新轮选中的 key 恰好之后再也不被问到，过时信息不产生影响 | 不刻意干预（这是真实情况），但统计"更新后被问到"的比例，E3 解释时要用 |
| B7 | session 第一轮就是更新，或连续多次更新同一个 key | 允许；值池要够大，每次更新都抽未用过的值 |
| B8 | 词表不够大，更新时抽不到新值 | 词表规模 ≥ 需求 × 3；抽不到时抛异常而不是静默复用 |

### C. 打分器

| # | 问题 | 处理 |
| --- | --- | --- |
| C1 | 大小写、标点、弯引号、连字符导致漏判 | `normalize` 同时作用于回答和别名 |
| C2 | 子串匹配误判（"Oak" 命中 "Oakland"） | 词边界匹配（两端补空格） |
| C3 | 日期写法多样（"the 14th"、"11/14"） | 别名表覆盖常见英文写法；数字日期格式不收（会和其他数字冲突）；人工标注时统计漏判，按需补别名 |
| C4 | "原来是 X，现在是 Y" 应算答对 | 当前值命中即答对，不管过时值是否也出现 |
| C5 | 策略可能学会"新旧都提"来骗分 | 记录 `hedged`；报告里给出各方法的 hedged 比例；如果某方法明显偏高，加一个小惩罚项做敏感性分析 |
| C6 | 用户自己刚说的值被模型复述，被算成无关提及 | 排除本轮用户输入里出现的值（§3.6） |
| C7 | 更新轮模型复述新值 | 新值属于 target_key，本来就排除 |
| C8 | 相关记录被提到也计数，导致下次被当作"被提及很多次"而惩罚加重 | n_m 只在无关提及时增加 |
| C9 | 模型提到一条从未注入、但用户早先在对话里说过的值 | 算无关提及（是模型主动提的），但在 mentions 里标来源 "history"，行为模型用这个特征 |

### D. 检索

| # | 问题 | 处理 |
| --- | --- | --- |
| D1 | `BM25Okapi` 在小语料上，词出现在一半以上文档时 IDF 变负，分数异常 | 用 `BM25Plus` |
| D2 | 新旧版本句子几乎一样，平分时排序有系统偏向（总是旧的或总是新的在前） | 用 id 的 sha256 打破平分；统计"旧版本排在新版本前面"的比例，应接近一半 |
| D3 | 训练时现场生成的脚本没有预计算 embedding，每步跑一次 MiniLM 太慢 | 轮文本由模板 + 槽位组成，种类有限：生成世界时对**全部模板实例化结果**预计算 embedding，按文本 sha256 查表；查不到才现场算并缓存 |
| D4 | 第一次运行时 sentence-transformers 要从 HuggingFace 下载模型 | 在 Mac 上第一次 `gen_world.py` 时下载，之后读本地缓存；固定模型名 `sentence-transformers/all-MiniLM-L6-v2` |
| D5 | Apple Silicon 上 torch 用 MPS 时 embedding 结果和 CPU 有微小差异 | embedding 只在生成世界时算一次并存盘，之后都读文件；强制 `device="cpu"` |
| D6 | 持久模式下排除已在上下文的记录后，候选不足 3 条 | 空位补齐，特征全 0，动作位忽略 |

### E. 环境与 RL

| # | 问题 | 处理 |
| --- | --- | --- |
| E1 | 把 session 结束标成 truncated，SB3 会用 bootstrap，值估计错 | `terminated=True, truncated=False` |
| E2 | 8 个动作里很多是重复的（空位导致）；DQN 浪费容量 | 接受；文档里说明；评估时把等价动作合并统计 |
| E3 | SB3 DQN 默认 `learning_starts` 太小，早期噪声大 | 设 `learning_starts=5000` |
| E4 | 不同 γ 的回报不可比 | 评估一律报不折扣总回报（§1.6） |
| E5 | 只给 γ=0.9 调参，其他 γ 吃亏 | 原文档已规定：只调一次、所有 γ 沿用；报告里说明 |
| E6 | DQN γ=0 和 LinUCB 的差别不只是 γ，还有探索方式和模型容量 | **主结论只基于两组对比**：DP 对贪心（同一模型，无学习误差）、DQN γ=0.9 对 DQN γ=0（同网络、同数据、同调参，只差 γ）。LinUCB 和风险敏感 bandit 只作为"已有方法的代表"放进表 1，不用来论证长期规划的价值（见 §3.13） |
| E7 | VecEnv 里 8 个环境用同一个 seed，生成相同的脚本序列 | 每个子环境 seed = base_seed × 100 + i |
| E8 | 观测里泄漏了生成器标注（superseded_by、relevant_ids） | features.py 不接收这些字段（函数签名就拿不到）；测试里随机改这些字段，观测必须不变 |
| E9 | 观测越界或 NaN（除以 0） | 归一化分母都有下限；`_obs` 里 clip + assert |
| E10 | 训练中 EvalCallback 用了测试用户 | eval 环境只用验证用户；测试用户只在 `evaluate.py` 里出现 |
| E11 | PPO 在短 episode、离散小动作空间上不稳定 | n_steps 2048 已覆盖很多 episode；5 个种子；如果明显不如 DQN，如实报告 |
| E12 | torch 多线程和多进程并行抢 CPU，反而更慢 | 每个进程 `torch.set_num_threads(1)`，进程数 = 核数 |
| E13 | Mac 训练中途睡眠 | 用 `caffeinate -i python ...` 运行长任务 |

### F. 模拟器（行为模型）

| # | 问题 | 处理 |
| --- | --- | --- |
| F1 | 之后想加特征，但日志里没存够信息，只能重新调 LLM | 日志存状态快照（§2.5） |
| F2 | 日志特征和模拟器特征用了两份代码，悄悄不一致 | 只有一个 `outcome_features`，测试里用同一快照对比两条路径 |
| F3 | 学出来的策略去到日志没覆盖的状态（例如一开始就全部注入、上下文里堆 10 条） | 行为策略 70% 均匀随机已经覆盖大部分；统计关键特征（上下文条数、被提及次数）的覆盖范围；超出范围时 sim 里裁剪到覆盖上限；E6 是最终检验 |
| F4 | "用了旧值"样本很少，逻辑回归校准差 | 看校准曲线；必要时合并稀疏特征或加交互项；梯度提升树对照（原文档已要求） |
| F5 | 提及事件独立性假设不成立（模型一次提到多条） | 统计日志里每轮多条提及的联合频率，和独立假设下的期望比较；偏差大就加"上下文 memory 数"交互项 |
| F6 | 一次性模式下的影响只通过模型自己的旧回答传递，特征里没有 | 提及模型特征里有"此前被模型提及过、且仍在历史里" |
| F7 | 模拟器在训练用户上拟合，测试用户的分布不同 | 用户之间结构相同（同一生成器），差异只在具体值；在验证用户的日志上检查（收集日志时额外跑 200 个验证用户 session，只用于这个检查） |

### G. DP 与小实例

| # | 问题 | 处理 |
| --- | --- | --- |
| G1 | 抽象状态到行为模型特征的映射写错，DP 用的概率和模拟器不一致 | "采样估计值与递归精确值一致"这条检查专门抓这个（原文档已列） |
| G2 | 状态数爆炸、内存不够 | 实际可达状态远少于上界；先跑 T=4 看缓存大小，再估算 T=10 |
| G3 | 浮点误差让 T=1 时 V^DP 和 V^greedy 有 1e-12 的差 | 检查用 `math.isclose(..., abs_tol=1e-9)` |
| G4 | DP 知道哪条是过时的，学习策略不知道，比较不公平 | 原文档已说明：Δ 只用 DP 与贪心对比 |
| G5 | 小实例的过时比例定义和完整规模不一致 | 两边都按 §1.5 定义 |

### H. 统计与实验

| # | 问题 | 处理 |
| --- | --- | --- |
| H1 | 不同方法在不同 session 上评估，差值噪声大 | 全部方法用同一组 2,000 个测试脚本；配对差值 |
| H2 | 真实 LLM 评估时，同一脚本在不同方法下的回答随机性不配对 | 接受；E6 每方法 300 session，报告置信区间；排序一致性用 Spearman |
| H3 | 多次比较（很多方法 × 很多设置） | 主结论只基于预先写定的几组比较（H1–H4）；其他标为探索性 |
| H4 | 阈值方法在测试集上调参 | τ 只在验证用户上调 |
| H5 | 一个配置文件展开成很多 run，某个 run 失败后不知道 | `train.py --sweep` 结束时打印每个 run 的状态表；失败的写进 `runs/_failed.txt` |

### I. 工程与复现

| # | 问题 | 处理 |
| --- | --- | --- |
| I1 | 用内置 `hash()` 做缓存 key 或 ID，跨进程不一致 | 一律 sha256（§1.1） |
| I2 | dataclass 里用可变默认值 | 全部 `field(default_factory=...)`；frozen dataclass 里的集合用 frozenset / tuple |
| I3 | parquet 里的嵌套列表、空列表类型推断出错 | 用显式 pyarrow schema 写入 |
| I4 | 异步代码在测试里难测 | LLM 客户端定义成 Protocol；测试用 `FakeLLM`（按规则返回固定回答），runner、打分、日志全链路都能离线测 |
| I5 | sklearn / torch 版本变化让 pickle 读不出来 | `requirements.lock`；模型文件旁边存版本号 |
| I6 | `data/`、`runs/` 不进 git，换机器后找不到 | 全部可由脚本重新生成；LLM 缓存 SQLite 单独备份到网盘（它是唯一花钱才能得到的东西） |
| I7 | 结果目录被覆盖 | 每个 run 目录名带时间戳和配置哈希；`--overwrite` 才允许覆盖 |
| I8 | 改了代码但用旧日志拟合，不知道哪个版本 | 日志里有 `git_commit` 和 `config_hash`；fit_sim 时检查并打印 |

### J. 研究层面

| # | 问题 | 处理 |
| --- | --- | --- |
| J1 | E0 不过关：模型几乎不受过时或无关 memory 影响 | 按顺序：换 SYSTEM_V2 → 让过时值更隐蔽（不出现"updated"这种词）→ 换更弱的模型；每次只改一处，跑 E0 的子集（每条件 20 个） |
| J2 | bandit 和长期策略没有差距 | 如实报告，解释原因；H4 的对照仍然有意义 |
| J3 | 模拟器和真实 LLM 排序不一致 | E6 报告 Spearman；不一致之处写进局限性 |
| J4 | 时间不够 | 可砍的顺序：E7 → PPO → 敏感性分析 → 一次性模式的 E1/E2（保留 E3/E5 的 Δ 曲线）。**不能砍**：摊销贪心和 R_behavior 下的 Δ，它们是核心论点的证据 |
| J5 | 评审质疑"长期结构是奖励函数造出来的"：token 成本按轮计费、β 逐次加重都是人为设计，即使 LLM 不受历史影响 bandit 也注定会输 | Δ 按奖励分项拆开；在 R_notok 和 R_behavior 下重新求解 Δ；主结论以行为驱动的 Δ 为准（§3.13） |
| J6 | 评审质疑"不需要 RL，把成本按剩余轮数摊销就行" | 摊销贪心作为基线，Δ_amort = V^DP − V^amortized-greedy 进入主结论；如果它接近 0，就如实写成"摊销即可" |
| J7 | Δ 是在拟合的行为模型上算出来的，E6 只验证方法排名，不能直接说明 Δ 本身在真实模型上成立 | 报告里明确区分"模拟器上的 Δ"和"真实模型上的排名"；E6 里额外报 DQN γ=0.9 与 γ=0、摊销贪心的真实回报差值，作为 Δ 在真实模型上的近似证据 |
| J8 | 世界合成、写入脚本化，外部有效性有限 | 写进局限性；有余力时做 LongMemEval 回放（原文档已列） |

---

## 5. 测试清单

和代码一起写完，`pytest` 全绿之后才开始第一次真实调用。

| 文件 | 内容 |
| --- | --- |
| `test_generator.py` | 同 seed 完全一致；别名按词边界不重叠；闲聊不含 value；answer_value 是当时最新值；split 不重叠；值池不复用 |
| `test_scorer.py` | ≥ 30 条手写回答：答对、只用旧值、新旧都提（correct + hedged）、无关提及、用户自己说的值、大小写标点、"Nov 1" vs "Nov 14"、Markdown 包裹 |
| `test_reward.py` | n_m 先用后加；token 项两种模式；非提问轮无质量项 |
| `test_prompt.py` | 持久模式 memory 留在历史；一次性模式历史里没有 `<memory>`；两种模式本轮消息完全相同 |
| `test_state.py` | 持久模式 in_context 只增不减；一次性模式每轮重置；更新轮写入新记录；conflict_flag |
| `test_retriever.py` | 长度恒为 3；exclude 生效；平分打破确定；旧版本可被检索到 |
| `test_env.py` | 观测在 [−1, 1]、无 NaN；mask_history 后历史位为 0；改生成器标注不影响观测；空位动作不注入；episode 长度 = T；gymnasium `check_env` 通过 |
| `test_backends.py` | 给定同一组 (answer, mentioned_ids)，live 和 sim 的奖励与新状态完全相同 |
| `test_runner.py` | 用 FakeLLM 跑 10 个 session：日志行数、字段齐全；模拟一次失败 session 被丢弃；断点续跑跳过已完成 |
| `test_llm_cache.py` | 相同 (messages, session, turn) 命中；不同 session 不命中 |
| `test_behavior_model.py` | 在合成日志上拟合能恢复已知的概率方向；日志路径与模拟路径特征一致 |
| `test_dp.py` | 原文档"必须成立的关系"全部 5 条（第 4 条在小实例上用短训练近似，标 `@pytest.mark.slow`）；另加：V^DP ≥ V^amortized-greedy；一次性模式下摊销贪心与贪心的动作完全相同；四个分项价值之和等于总价值；R_behavior 下若提及概率与历史无关则 Δ = 0 |

---

## 6. 写代码的顺序与运行顺序

### 6.1 写代码（一口气写完，按依赖顺序）

1. `pyproject.toml`、`configs/*.yaml`、`memgate/config.py`（pydantic 模型）
2. `schema.py`
3. `world/pools.py`、`templates.py`、`generator.py`
4. `retrieval/retriever.py`
5. `harness/scorer.py`（含 `compute_reward`）、`prompt.py`、`llm_client.py`（含 FakeLLM、缓存）
6. `env/state.py`、`features.py`、`backend_live.py`、`backend_sim.py`、`gate_env.py`
7. `harness/runner.py`
8. `sim/outcome_features.py`、`behavior_model.py`
9. `planning/abstract.py`、`dp.py`、`greedy.py`
10. `policies/heuristics.py`、`linucb.py`、`sb3.py`
11. `analysis/stats.py`、`plots.py`
12. `scripts/*.py`（gen_world、run_pilot、collect_logs、fit_sim、solve_dp、train、evaluate、make_figures）
13. `tests/` 全部（与对应模块同时写）
14. `configs/exp/e0_pilot.yaml` … `e7_model.yaml`

写完后：`ruff check`、`pytest`（不含 slow）全绿，再进入运行。

### 6.2 运行（有数据依赖，必须按顺序）

代码可以一次写完，但运行有先后：模拟器要等日志，DP 要等模拟器。

| 顺序 | 命令 | 依赖 | 产出 |
| --- | --- | --- | --- |
| R1 | `gen_world.py --seed 0` | 无 | `data/world_s0/` |
| R2 | `run_pilot.py` | R1 | `runs/e0/`，判断 G0 |
| R3 | `collect_logs.py`（两种模式） | R1，G0 通过 | `data/logs/` |
| R4 | 人工标注 200 条（E0 + 日志） | R2、R3 | 打分器一致率 |
| R5 | `fit_sim.py` | R3 | `runs/sim/` |
| R6 | `solve_dp.py --reward R_full R_notok R_behavior`（每个变体都算 DP、贪心、摊销贪心，并输出分项价值） | R5 | `runs/e3/` |
| R7 | `train.py --sweep e2/e1/e4/e5` | R5 | `runs/...` |
| R8 | `evaluate.py --backend sim` | R7 | `results/sim_eval.parquet` |
| R9 | `evaluate.py --backend live` | R7 | `results/live_eval.parquet` |
| R10 | `make_figures.py` | 全部 | `figures/` |

---

## 7. 时间线与关卡

| 日期 | 做什么 | 关卡 |
| --- | --- | --- |
| 10/27–10/28 | §6.1 全部代码 + 测试写完 | 测试全绿 |
| 10/29 | R1、R2 | **G0**：无关 memory 后续重复提及率 ≥ 10%，或纠正后旧值使用率 ≥ 10%；不过按 J1 处理 |
| 10/30 | 提交 1 页 proposal | **G1** |
| 10/31–11/6 | R3（后台）、R4 | **G2**：打分器一致率 答对/旧值 ≥ 95%、无关提及 ≥ 90%；日志收集完成 |
| 11/7–11/13 | R5、R6、第一版 γ 扫描（1 种子） | **G3**：模拟器三项验证通过；DP 5 条关系全部成立 |
| 11/14–11/20 | R7（含 R_behavior 下 DQN γ ∈ {0, 0.9} × 5 种子）、R8 | **G4**：表 1、图 2、图 3 可生成，带置信区间；Δ、Δ_amort、行为驱动的 Δ 三个数都已算出 |
| 11/21–11/27 | R9、敏感性分析、可选 E7 | **G5**：排序一致性算出 |
| 11/28–12/7 | R10、报告、slides、Gradescope | **G6**：干净 clone 能复现全部图表（真实调用走缓存） |
| 12/10 17:00 | 提交最终报告 | |

代码是一次写完的，所以 10/31 之后的时间几乎全部用在跑实验、检查和写报告上；如果某个关卡暴露出代码问题，修完重跑对应的 R 步骤。

---

## 8. 写代码之前还要确认的事

1. **DeepSeek 文档**：当前的 temperature 说明（A6）、usage 里缓存字段的名字（A9）、价格和优惠时段（A11）。
2. **预算上限**：`budget_usd` 设多少。
3. **课程要求**：proposal 和最终报告的格式、页数是否和原文档一致。（托管 API 已确认可以用：课程 guideline 写的是 "You are free to use existing software and packages"。）
4. **§1.2 的 8 条改动**和 **§1.4、§1.5 的规模与定义**：没有异议就按这里写。
