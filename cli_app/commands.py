from __future__ import annotations

from pathlib import Path

from Agent_Assistant.cli_app.client import call_core
from Agent_Assistant.cli_app.output import print_help, print_json, print_status
from Agent_Assistant.cli_app.state import CliState


def parse_on_off(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"on", "true", "yes", "y", "1"}:
        return True
    if normalized in {"off", "false", "no", "n", "0"}:
        return False
    return None


def set_bool_option(name: str, raw_value: str, current: bool) -> bool:
    if not raw_value:
        new_value = not current
    else:
        parsed = parse_on_off(raw_value)
        if parsed is None:
            print(f"usage: /{name} on|off")
            return current
        new_value = parsed
    print(f"{name}: {'on' if new_value else 'off'}")
    return new_value


async def handle_command(line: str, state: CliState) -> bool:
    command, _, raw_value = line.partition(" ")
    value = raw_value.strip()

    if command in {"/exit", "/quit", "/q"}:
        return False
    if command == "/help":
        print_help()
    elif command == "/ping":
        result = await call_core("ping", {"message": "ping"})
        print_json(result)
    elif command == "/workspace":
        if value:
            state.workspace = value
        print(f"workspace: {Path(state.workspace).resolve()}")
    elif command == "/rag":
        state.enable_rag = set_bool_option("rag", value, state.enable_rag)
    elif command == "/verbose":
        state.verbose = set_bool_option("verbose", value, state.verbose)
    elif command == "/json":
        state.raw_json = set_bool_option("json", value, state.raw_json)
    elif command == "/stream":
        state.stream = set_bool_option("stream", value, state.stream)
    elif command == "/yes":
        state.auto_approve = set_bool_option("yes", value, state.auto_approve)
    elif command == "/steps":
        handle_steps(value, state)
    elif command == "/status":
        print_status(state)
    else:
        print(f"unknown command: {command}")
        print("type /help for commands")
    return True


def handle_steps(value: str, state: CliState) -> None:
    if not value:
        print(f"steps: {state.max_steps}")
        return
    try:
        steps = int(value)
    except ValueError:
        print("usage: /steps 1-30")
        return
    if 1 <= steps <= 30:
        state.max_steps = steps
        print(f"steps: {state.max_steps}")
    else:
        print("usage: /steps 1-30")
