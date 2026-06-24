from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Agent_Asistant.cli_app.state import CliState

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

ASCII_LOGO = r"""
   ___                    _        ___        _      _              _
  / _ \  __ _  ___  _ __ | |_     /   \ ___  (_) ___| |_ __ _ _ __ | |_
 / /_)/ / _` |/ _ \| '_ \| __|   / /\ /(_-<  | |(_-<| __/ _` | '_ \|  _|
/ ___/ | (_| |  __/| | | | |_   / /_// /__/  | |/__/| || (_| | | | | |_
\/      \__, |\___||_| |_|\__| /___,' |___/ _/ ||___| \__\__,_|_| |_|\__|
        |___/                              |__/
"""


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def color(text: str, value: str) -> str:
    return f"{value}{text}{RESET}"


def format_ms(elapsed_ms: Any) -> str:
    try:
        ms = int(elapsed_ms)
    except (TypeError, ValueError):
        return "unknown"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.2f}s"


def usage_line(data: dict[str, Any]) -> str:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return "tokens 不可用"
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_tokens = int(usage.get("cache_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    estimated = bool(usage.get("estimated"))
    suffix = " 估算" if estimated else ""
    return (
        f"tokens in={input_tokens} out={output_tokens} "
        f"cache={cache_tokens} total={total_tokens}{suffix}"
    )


def print_run_result(data: dict[str, Any]) -> None:
    status = str(data.get("status", ""))
    status_color = GREEN if status == "success" else RED
    print()
    print(color(f"run {data.get('run_id', '')}", CYAN))
    print(f"status {color(status, status_color)}")
    print(f"steps  {data.get('steps', '')}")
    print(f"time   {format_ms(data.get('elapsed_ms'))}")
    print(usage_line(data))
    print(f"events {data.get('events_path', '')}")
    result = str(data.get("result") or "").strip()
    if result:
        print()
        print(result)
    print()
    print(color("已完成", GREEN) if status == "success" else color("失败", RED))


def print_run_summary(data: dict[str, Any]) -> None:
    status = str(data.get("status", ""))
    status_color = GREEN if status == "success" else RED
    print()
    print()
    print(color(f"run {data.get('run_id', '')}", CYAN))
    print(f"status {color(status, status_color)}")
    print(f"steps  {data.get('steps', '')}")
    print(f"time   {format_ms(data.get('elapsed_ms'))}")
    print(usage_line(data))
    print(f"events {data.get('events_path', '')}")
    print(color("已完成", GREEN) if status == "success" else color("失败", RED))


def print_banner(state: CliState) -> None:
    print()
    print(
        f"{BOLD}Agent_Asistant{RESET}  "
        f"127.0.0.1  sess-{state.session_id}  {color('就绪', GREEN)}"
    )
    print(color(ASCII_LOGO, CYAN))
    print(
        f"{DIM}输入消息开始对话 · 已开启流式输出 · /help 查看命令 · "
        f"Ctrl+C 或 /exit 退出{RESET}"
    )
    print()
    print(f"工作区 {Path(state.workspace).resolve()}")
    rag_status = "开启" if state.enable_rag else "关闭"
    verbose_status = "开启" if state.verbose else "关闭"
    print(f"RAG {rag_status} · 详细模式 {verbose_status}")
    print()


def print_help() -> None:
    print(
        "\n".join(
            [
                "命令：",
                "  /help                    显示帮助",
                "  /ping                    检查 Core 是否在线",
                "  /workspace [path]        查看或切换工作区",
                "  /rag on|off              开启或关闭本地 RAG 工具",
                "  /verbose on|off          开启或关闭详细回答",
                "  /json on|off             开启或关闭原始 JSON 输出",
                "  /stream on|off           开启或关闭 token 流式输出",
                "  /yes on|off              开启或关闭写文件自动审批",
                "  /steps [1-30]            查看或设置最大 Agent 步数",
                "  /status                  显示当前 CLI 状态",
                "  /exit                    退出",
                "",
                "其他输入会作为 agent.run 任务发送给 Core。",
            ]
        )
    )


def print_status(state: CliState) -> None:
    print(f"工作区: {Path(state.workspace).resolve()}")
    print(f"max_steps: {state.max_steps}")
    print(f"写文件自动审批: {'开启' if state.auto_approve else '关闭'}")
    print(f"RAG: {'开启' if state.enable_rag else '关闭'}")
    print(f"context_max_chars: {state.context_max_chars}")
    print(f"详细模式: {'开启' if state.verbose else '关闭'}")
    print(f"JSON 输出: {'开启' if state.raw_json else '关闭'}")
    print(f"流式输出: {'开启' if state.stream else '关闭'}")
