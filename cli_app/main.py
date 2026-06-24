from __future__ import annotations

import asyncio
from typing import NoReturn

from Agent_Asistant.cli_app.client import call_core, run_agent
from Agent_Asistant.cli_app.output import print_json, print_run_result
from Agent_Asistant.cli_app.parser import build_parser, build_state
from Agent_Asistant.cli_app.repl import interactive_loop
from Agent_Asistant.cli_app.state import CliState


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "chat"
    if command == "chat":
        state = CliState() if args.command is None else build_state(args)
        asyncio.run(interactive_loop(state))
    elif command == "tui":
        from Agent_Asistant.cli_app.tui import run_tui

        state = build_state(args)
        run_tui(state)
    elif command == "ping":
        result = asyncio.run(call_core("ping", {"message": args.message}))
        print_json(result)
    elif command == "run":
        state = build_state(args)
        result = asyncio.run(run_agent(args.goal, state))
        if args.json:
            print_json(result)
        else:
            print_run_result(result)
    else:
        die(f"unknown command: {command}")
