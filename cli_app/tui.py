from __future__ import annotations

import asyncio
import curses
import locale
import textwrap
import time
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any

from Agent_Assistant.cli_app.client import stream_agent
from Agent_Assistant.cli_app.output import format_ms, usage_line
from Agent_Assistant.cli_app.state import CliState

MAX_LINES = 500


def cell_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def fit_cells(text: str, max_width: int) -> str:
    used = 0
    chars: list[str] = []
    for char in text:
        char_width = cell_width(char)
        if used + char_width > max_width:
            break
        chars.append(char)
        used += char_width
    return "".join(chars)


class TuiApp:
    def __init__(self, stdscr: curses.window, state: CliState) -> None:
        self.stdscr = stdscr
        self.state = state
        self.lines: deque[str] = deque(maxlen=MAX_LINES)
        self.input_text = ""
        self.input_cursor = 0
        self.input_offset = 0
        self.status = "就绪"
        self.running = False
        self.last_result: dict[str, Any] | None = None

    def run(self) -> None:
        locale.setlocale(locale.LC_ALL, "")
        curses.curs_set(1)
        self.stdscr.keypad(True)
        curses.mousemask(curses.BUTTON1_CLICKED)
        curses.mouseinterval(0)
        self.stdscr.timeout(100)
        self._init_colors()
        self.add_line("Agent_Assistant TUI 已就绪。输入 /help 查看命令。")

        while True:
            self.render()
            try:
                key = self.stdscr.get_wch()
            except curses.error:
                continue
            if key in {3, 4}:
                return
            if key in {"\n", "\r", curses.KEY_ENTER}:
                text = self.input_text.strip()
                self.input_text = ""
                self.input_cursor = 0
                self.input_offset = 0
                if not text:
                    continue
                if text in {"/exit", "/quit", "/q"}:
                    return
                if text.startswith("/"):
                    self.handle_command(text)
                    continue
                self.run_turn(text)
                continue
            if key in {curses.KEY_BACKSPACE, "\b", "\x7f"}:
                self.delete_before_cursor()
                continue
            if key == curses.KEY_DC:
                self.delete_at_cursor()
                continue
            if key == curses.KEY_LEFT:
                self.input_cursor = max(0, self.input_cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                self.input_cursor = min(len(self.input_text), self.input_cursor + 1)
                continue
            if key == curses.KEY_HOME:
                self.input_cursor = 0
                continue
            if key == curses.KEY_END:
                self.input_cursor = len(self.input_text)
                continue
            if key == curses.KEY_MOUSE:
                self.handle_mouse()
                continue
            if key == curses.KEY_RESIZE:
                continue
            if isinstance(key, str) and key.isprintable():
                self.insert_text(key)

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)

    def color(self, pair: int) -> int:
        if not curses.has_colors():
            return curses.A_NORMAL
        return curses.color_pair(pair)

    def add_line(self, text: str = "") -> None:
        width = max(20, self.stdscr.getmaxyx()[1] - 4)
        wrapped = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
        for line in wrapped:
            self.lines.append(line)

    def append_text(self, text: str) -> None:
        if not text:
            return
        parts = text.splitlines(keepends=True)
        for part in parts:
            if not self.lines:
                self.lines.append("")
            if part.endswith("\n"):
                self.lines[-1] += part[:-1]
                self.lines.append("")
            else:
                self.lines[-1] += part

    def render(self) -> None:
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        if height < 8 or width < 40:
            self.stdscr.addstr(0, 0, "终端窗口过小")
            self.stdscr.refresh()
            return

        header = (
            f" Agent_Assistant  127.0.0.1  sess-{self.state.session_id}  "
            f"{self.status} "
        )
        self.stdscr.addnstr(0, 0, header.ljust(width), width, self.color(5))
        self._draw_logo(width)
        self._draw_messages(height, width)
        self._draw_footer(height, width)
        self.stdscr.refresh()

    def _draw_logo(self, width: int) -> None:
        logo = [
            "    AGENT_ASSISTANT",
            "    本地 Agent Runtime / JSON-RPC / 流式输出",
        ]
        for idx, line in enumerate(logo, start=1):
            self.stdscr.addnstr(idx, 1, line[: width - 2], width - 2, self.color(1))

    def _draw_messages(self, height: int, width: int) -> None:
        top = 4
        bottom = height - 4
        visible = list(self.lines)[-(bottom - top) :]
        for row, line in enumerate(visible, start=top):
            self.stdscr.addnstr(row, 1, line, width - 2)

    def _draw_footer(self, height: int, width: int) -> None:
        help_text = " 输入消息 - 回车发送 - /help 帮助 - /exit 退出 "
        self.stdscr.addnstr(height - 3, 1, help_text.ljust(width - 2), width - 2, self.color(3))
        input_width = max(1, width - 4)
        self.sync_input_offset(input_width)
        visible_input = fit_cells(self.input_text[self.input_offset :], input_width)
        self.stdscr.addnstr(height - 2, 1, " " * (width - 2), width - 2)
        self.stdscr.addnstr(height - 2, 1, "> " + visible_input, width - 2)
        cursor_cells = cell_width(self.input_text[self.input_offset : self.input_cursor])
        cursor_x = min(width - 2, 3 + cursor_cells)
        self.stdscr.move(height - 2, cursor_x)

    def sync_input_offset(self, input_width: int) -> None:
        self.input_cursor = max(0, min(self.input_cursor, len(self.input_text)))
        self.input_offset = max(0, min(self.input_offset, self.input_cursor))
        while self.input_offset > self.input_cursor:
            self.input_offset -= 1
        while cell_width(self.input_text[self.input_offset : self.input_cursor]) >= input_width:
            self.input_offset += 1

    def insert_text(self, text: str) -> None:
        self.input_text = (
            self.input_text[: self.input_cursor] + text + self.input_text[self.input_cursor :]
        )
        self.input_cursor += len(text)

    def delete_before_cursor(self) -> None:
        if self.input_cursor == 0:
            return
        self.input_text = (
            self.input_text[: self.input_cursor - 1] + self.input_text[self.input_cursor :]
        )
        self.input_cursor -= 1
        self.input_offset = min(self.input_offset, self.input_cursor)

    def delete_at_cursor(self) -> None:
        if self.input_cursor >= len(self.input_text):
            return
        self.input_text = (
            self.input_text[: self.input_cursor] + self.input_text[self.input_cursor + 1 :]
        )

    def handle_mouse(self) -> None:
        try:
            _, x, y, _, button_state = curses.getmouse()
        except curses.error:
            return
        height, width = self.stdscr.getmaxyx()
        input_row = height - 2
        if y != input_row or not button_state & curses.BUTTON1_CLICKED:
            return
        input_start = 3
        input_width = max(1, width - 4)
        self.sync_input_offset(input_width)
        target_cells = max(0, min(input_width, x - input_start))
        self.input_cursor = self.cursor_from_cells(target_cells)

    def cursor_from_cells(self, target_cells: int) -> int:
        cells = 0
        for index, char in enumerate(self.input_text[self.input_offset :], start=self.input_offset):
            char_width = max(1, cell_width(char))
            if target_cells < cells + (char_width / 2):
                return index
            cells += char_width
            if target_cells < cells:
                return index + 1
        return len(self.input_text)

    def handle_command(self, command: str) -> None:
        if command == "/help":
            self.add_line("命令: /help /status /verbose on|off /rag on|off /exit")
        elif command == "/status":
            self.add_line(f"工作区: {Path(self.state.workspace).resolve()}")
            self.add_line(f"RAG: {'开启' if self.state.enable_rag else '关闭'}")
            self.add_line(f"详细模式: {'开启' if self.state.verbose else '关闭'}")
            self.add_line(f"流式输出: {'开启' if self.state.stream else '关闭'}")
        elif command.startswith("/verbose"):
            self.state.verbose = self._read_on_off(command, self.state.verbose)
            self.add_line(f"详细模式: {'开启' if self.state.verbose else '关闭'}")
        elif command.startswith("/rag"):
            self.state.enable_rag = self._read_on_off(command, self.state.enable_rag)
            self.add_line(f"RAG: {'开启' if self.state.enable_rag else '关闭'}")
        else:
            self.add_line(f"未知命令: {command}")

    def _read_on_off(self, command: str, current: bool) -> bool:
        parts = command.split(maxsplit=1)
        if len(parts) == 1:
            return not current
        value = parts[1].strip().lower()
        if value in {"on", "true", "yes", "1"}:
            return True
        if value in {"off", "false", "no", "0"}:
            return False
        return current

    def run_turn(self, text: str) -> None:
        self.add_line(f"> {text}")
        self.status = "运行中"
        self.running = True
        self.render()
        started = time.monotonic()
        try:
            result = asyncio.run(self._run_stream(text))
        except RuntimeError as exc:
            self.add_line(f"[error] {exc}")
            self.status = "error"
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if result is not None:
            self.last_result = result
            self.add_line("")
            self.add_line(
                f"已完成 {result.get('steps', '?')} 步，耗时 "
                f"{format_ms(result.get('elapsed_ms', elapsed_ms))}"
            )
            self.add_line(usage_line(result))
            self.add_line(f"事件日志 {result.get('events_path', '')}")
        self.status = "就绪"
        self.running = False

    async def _run_stream(self, text: str) -> dict[str, Any] | None:
        final_result: dict[str, Any] | None = None
        async for event in stream_agent(text, self.state):
            event_type = str(event.get("type", ""))
            if event_type == "run.started":
                self.add_line(f"运行 {event.get('run_id', '')}")
            elif event_type == "step.started":
                self.add_line(f"步骤 {event.get('step', '')}")
            elif event_type == "text.delta":
                self.append_text(str(event.get("text") or ""))
            elif event_type == "tool.started":
                self.add_line(f"工具 {event.get('tool_name', '')} 执行中")
            elif event_type == "tool.finished":
                self.add_line(f"工具 {event.get('tool_name', '')} 完成")
            elif event_type == "result":
                data = event.get("data")
                if isinstance(data, dict):
                    final_result = data
            self.render()
        return final_result


def run_tui(state: CliState) -> None:
    curses.wrapper(lambda stdscr: TuiApp(stdscr, state).run())
