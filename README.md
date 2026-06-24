# Agent Assistant

Agent_Assistant 是一个本地 Agent Runtime 演示项目，采用 CLI/Core 分离架构：

```text
CLI / TUI 客户端
  -> JSON-RPC 2.0 over NDJSON
  -> Core 常驻进程
  -> 协议校验与权限控制
  -> AgentRunner 主循环
  -> DeepSeek / OpenAI / Anthropic-compatible 模型适配
  -> 工具注册表
  -> 本地 RAG 检索
  -> 上下文压缩
  -> JSONL 事件追踪
```

## 项目结构

```text
Agent_Assistant/
|-- cli.py                  # CLI 入口
|-- core.py                 # Core 守护进程入口
|-- config.py               # .env 与运行配置
|-- cli_app/                # CLI、REPL、TUI 展示层
|-- runtime/                # Agent 主循环、上下文压缩、事件追踪
|-- llm_clients/            # DeepSeek/OpenAI/Anthropic 兼容模型适配
|-- tooling/                # 工作区工具与本地 RAG
`-- rpc/                    # JSON-RPC 协议模型与 NDJSON 传输层
```

## 在 Linux ECS 上运行

进入项目目录并准备 `.env`：

```bash
cd Agent_Assistant
cp env.example .env
```

支持两类配置：

- `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`
- `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL=https://api.deepseek.com`

启动 Core：

```bash
uv run python core.py
```

另开一个终端测试连接：

```bash
uv run python cli.py ping
```

## 交互模式

普通 CLI：

```bash
uv run python cli.py chat --workspace .
```

TUI 图形化终端界面：

```bash
uv run python cli.py tui --workspace .
```

TUI 基于 Python 标准库 `curses`，适合 Linux SSH 终端。它包含顶部状态栏、消息区、底部输入框、流式 token 输出、工具事件、耗时和 token 汇总。

交互命令示例：

```text
/help
/ping
/workspace .
/verbose on
/rag off
/exit
```

默认启用真流式输出：模型 delta 会从 Core 通过 NDJSON 事件实时转发到 CLI/TUI。如果需要旧的一次性响应模式：

```bash
uv run python cli.py chat --workspace . --no-stream
```

## 单次任务模式

```bash
uv run python cli.py run \
  --workspace /path/to/project \
  "阅读 README.md，并生成 notes/demo_summary.md"
```

查看原始 JSON-RPC 返回：

```bash
uv run python cli.py run --json --workspace . "总结项目结构"
```

需要更长回答：

```bash
uv run python cli.py run --verbose --workspace . "详细分析架构亮点"
```

禁用 RAG：

```bash
uv run python cli.py run --no-rag --workspace . "列出项目模块"
```

可信演示场景下自动批准写文件：

```bash
uv run python cli.py run --yes --workspace . "创建 notes/hello.md"
```
