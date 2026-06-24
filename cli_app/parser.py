from __future__ import annotations

import argparse

from Agent_Asistant.cli_app.state import CliState


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--yes", action="store_true", help="自动批准 write_file 写文件请求")
    parser.add_argument("--no-rag", action="store_true", help="禁用 rag_search 本地检索工具")
    parser.add_argument("--context-max-chars", type=int, default=24_000)
    parser.add_argument("--verbose", action="store_true", help="允许模型输出更详细的回答")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON 结果")
    parser.add_argument("--no-stream", action="store_true", help="关闭 token 流式输出")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-asistant",
        description="Agent_Asistant 本地 Agent Runtime 命令行",
    )
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="打开交互式 CLI")
    add_runtime_options(chat)

    tui = sub.add_parser("tui", help="打开 curses 图形化终端界面")
    add_runtime_options(tui)

    ping = sub.add_parser("ping")
    ping.add_argument("--message", default="ping")

    run = sub.add_parser("run")
    run.add_argument("goal")
    add_runtime_options(run)
    return parser


def build_state(args: argparse.Namespace) -> CliState:
    return CliState(
        workspace=str(args.workspace),
        max_steps=int(args.max_steps),
        auto_approve=bool(args.yes),
        enable_rag=not bool(args.no_rag),
        context_max_chars=int(args.context_max_chars),
        verbose=bool(args.verbose),
        raw_json=bool(args.json),
        stream=not bool(args.no_stream),
    )
