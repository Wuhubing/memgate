# memgate

在 LLM 固定不动的 agent harness 里训练一个 memory 准入控制器：每一轮决定哪些 memory 进入模型输入，目标是整个 session 的累计回报。核心问题是这个决策能否当作 bandit，还是必须当作 MDP 做长期规划。

- 执行计划：[plan.md](plan.md)
- LLM：DeepSeek API（OpenAI 兼容接口），通过 `openai` SDK 调用

## 快速开始

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # pyproject.toml 建好之后
cp .env.example .env             # 填入 DEEPSEEK_API_KEY，.env 不进 git
```

## 密钥

API key 只放在本地 `.env` 或 shell 环境变量里，代码通过 `os.environ["DEEPSEEK_API_KEY"]` 读取。`.env`、`data/`、`runs/`、`results/` 都在 `.gitignore` 里。提交前可以用 `git diff --cached | grep -i "sk-"` 检查一遍。
