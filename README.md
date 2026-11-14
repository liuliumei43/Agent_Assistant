# Agent Assistant

一个可审阅文件修改、连续对话的本地代码助手。Python 3.12，CLI/TUI 与 Core 分离，通过 JSON-RPC + NDJSON 传输事件；支持 Chat Completions 与 Anthropic Messages 协议。

## v0.2.0 更新

- **会话记忆**：SQLite 保存成功完成的对话，`--session` 恢复，`/new` 新建；会话绑定工作区和模型配置。
- **客户端审批**：CLI/TUI 显示真实 diff，异步批准或拒绝；等待期间 Core 仍可响应其他请求，断连取消待审批动作。
- **安全编辑**：`read_file` 返回文件 hash；`apply_patch` 精确替换；写入前再次校验 hash，避免覆盖期间发生的新修改。
- **文件定位**：按行/列分页读取、文字搜索、glob、带行号的轻量 TF-IDF；索引复用未变化的文件。
- **模型协议**：保留 DeepSeek `reasoning_content` 和 Anthropic thinking/signature 块；检查终止状态，拒绝半截工具参数，流内错误和断流不再伪装成功。
- **统一执行循环**：流式和非流式模型调用共享工具与会话逻辑；每个 run 独立预算和 UUID 日志目录。
- **显式工作流**：`--skill repo-review` 加载内置代码审查 Skill。工具 schema 从 Pydantic 模型生成。

## 安装与启动

安装 Python 3.12 和 [uv](https://docs.astral.sh/uv/)，然后：

```bash
git clone https://github.com/liuliumei43/Agent_Assistant.git
cd Agent_Assistant
uv sync --frozen --dev
cp env.example .env
```

编辑 `.env`，选择 provider、填入对应 Key，并确认模型在自己账号可用。程序不再猜测默认模型名。原生 provider 会使用各自默认地址；兼容网关使用 `custom` 显式配置。

```dotenv
AGENT_ASSISTANT_PROVIDER=deepseek
DEEPSEEK_API_KEY=填写你的密钥
DEEPSEEK_MODEL=填写账号可用的模型名
```

启动 Core：

```bash
uv run agent-assistant-core
```

另开终端启动客户端：

```bash
uv run agent-assistant ping
uv run agent-assistant chat --workspace /absolute/path/to/project
```

亦可继续使用 `uv run python core.py` / `uv run python cli.py ...`。CLI 会将工作区转换为绝对路径，避免与 Core 启动目录混淆。

TUI（Linux/macOS 终端）：

```bash
uv run agent-assistant tui --workspace /absolute/path/to/project
```

写入审批时，CLI 输入 `y`；TUI 按 `y` 批准、`n` 拒绝，可用上下键或 PageUp/PageDown 浏览完整 diff。审批 5 分钟后自动拒绝。`--yes` 为用户显式开启的自动写入许可；默认关闭。

## 会话与单次任务

```bash
uv run agent-assistant chat --workspace /path/to/project --session interview-demo
uv run agent-assistant run --workspace /path/to/project --session interview-demo "找到认证失败的处理位置，并引用行号"
uv run agent-assistant run --workspace /path/to/project --session interview-demo "继续解释刚才的函数"
uv run agent-assistant run --workspace /path/to/project --skill repo-review "检查错误处理"
```

常用命令：`/help`、`/status`、`/new`、`/workspace path`、`/rag on|off`、`/stream on|off`、`/steps 1-30`、`/exit`。切换工作区会创建新会话。

`--no-stream` 只关闭模型 token 流式输出，CLI 仍使用事件连接接收审批。直接调用 `agent.run` 的非流式 RPC 没有审批 UI，未启用 `auto_approve` 时会拒绝写入。`--json` 或无交互 stdin 时默认拒绝审批；可信自动化可显式使用 `--yes`。

## 工具

| 工具 | 用途 |
| --- | --- |
| `list_dir` | 按深度列出文件，最多 200 个 |
| `glob_files` | 按文件名模式定位文件 |
| `search_text` | 字面搜索，返回路径、行号与片段 |
| `read_file` | 按行、列和字符上限读取，返回 sha256 与下一页位置 |
| `rag_search` | 文件级 TF-IDF 检索；最多 200 个文本文件，每文件前 64 KB |
| `write_file` | 创建文件；替换已有文件必须提供旧 sha256 |
| `apply_patch` | 精确替换唯一匹配的旧文本，提供旧 sha256，审批 diff |

文件工具共用路径边界检查，拒绝符号链接以及 `.git`、`.env`、`.runs`、`.agent-state` 等受保护路径。目录发现和搜索支持工作区根目录 `.agentignore`，每行一个相对路径 glob；不支持否定规则，也不等同完整 `.gitignore` 语义。

单文件读取上限为 2 MiB，写入上限为 512 KiB，单次审批 diff 上限为 100000 字符。读取被分页后可继续读取后续位置，不再统一丢弃文件尾部。

## 状态、错误与兼容范围

默认在项目安装目录的 `.agent-state/` 保存 `sessions.sqlite3` 和 `runs/<UUID>/events.jsonl`。可用 `AGENT_ASSISTANT_STATE_DIR` 指定私有目录。对话和工具输出可能包含源码，应按本地项目资料管理；这些文件均被 Git 忽略。

- 当前实现**成功对话的持久化**。中断或失败的本轮不会自动写入会话；文件修改若已完成，不会自动回滚或重放。完整任务 checkpoint/崩溃恢复仍在路线图中。
- `context_max_chars` 是字符预算，计入完整消息、工具参数、模型状态和工具 schema；不是精确 token 上限。旧的完整对话轮次可压缩，当前工具/推理轮次保持完整。若仍超预算，会明确失败，可缩小任务或提高参数。
- DeepSeek thinking 可设置 `AGENT_ASSISTANT_THINKING=enabled|disabled|default`。原生 Anthropic 的 thinking 参数随模型变化，本版支持默认行为或显式关闭，并保留服务端返回的 thinking/signature；不宣称所有模型特性均兼容。
- 同一模型配置下的流式和非流式使用同一校验规则。Chat Completions 必须有正常 finish reason 和 `[DONE]`；Anthropic 必须有完整 block、正常 stop reason 与 `message_stop`。
- Core 只绑定 loopback，面向受信任的本地单用户环境，没有远程身份认证。文件边界与 hash 检查不是对抗恶意并发进程的 OS 沙箱。
- 当前工具不执行 shell/测试；助手不能声称已运行用户仓库的测试。没有自动重试或自动重放写入。

## 无需 API Key 的演示

```bash
uv run python examples/offline_demo.py
```

演示使用脚本化模型，但调用真实的运行时、文件读取、diff、审批、patch 和 SQLite。流程为：读取临时配置 → 展示 diff → 演示审批器批准 → 验证文件 → 重建 Runner → 从同一会话继续。所有文件位于临时目录，不访问真实模型，也不修改用户项目。它验证工程链路，不代表真实模型任务成功率。

## 测试与开发

```bash
uv sync --frozen --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

测试包括协议帧、真实 loopback RPC、路径边界、审批拒绝/超时/断连、文件冲突、会话恢复和检索缓存。均使用虚构 Key、模拟模型响应和临时文件，无需付费 API。限制网络监听的沙箱需要允许本地回环连接才能运行 RPC 测试。

GitHub Actions 执行相同检查。真实 LLM 的质量、延迟和 token 开销需另用固定任务集多次评测；不把单元测试通过率作为 Agent 任务成功率。

## 架构

```text
CLI / TUI
  -> JSON-RPC + NDJSON events
  -> Core (loopback)
       -> ApprovalBroker (async request / response)
       -> AgentRunner (one execution loop)
            -> ModelClient (Chat Completions / Anthropic)
            -> ToolRegistry (Pydantic schemas)
            -> Workspace filesystem policy / hash-checked patch
            -> SessionStore (SQLite)
            -> Per-run JSONL events
```

## 下一阶段

1. 加入真实模型任务评测，比较检索、编辑与跨轮完成质量。
2. 增加 MCP 工具适配器；当前自定义 RPC 不是 MCP。
3. 增加可恢复任务状态与副作用账本，再考虑隔离的测试执行环境。
4. 根据检索基准决定是否引入 embedding、混合召回或 reranker。

参见 [更新记录](CHANGELOG.md) 与 [升级验证记录](docs/upgrade-validation.md)。
