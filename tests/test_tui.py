import curses

import pytest
from Agent_Assistant.cli_app.state import CliState
from Agent_Assistant.cli_app.tui import TuiApp


class Screen:
    def __init__(self, keys):
        self.keys = iter(keys)

    def getmaxyx(self):
        return 30, 100

    def get_wch(self):
        return next(self.keys)

    def nodelay(self, enabled):
        pass

    def timeout(self, value):
        self.restored_timeout = value


@pytest.mark.parametrize("key,expected", [("y", True), ("n", False)])
async def test_tui_can_scroll_full_diff_and_restore_history(key, expected):
    screen = Screen([curses.KEY_PPAGE, key])
    app = TuiApp(screen, CliState())
    app.lines.append("existing history")
    rendered = []
    app.render = lambda: rendered.append((len(app.lines), app.view_offset))
    decision = await app.approve(
        {"path": "file.txt", "diff": "\n".join(f"+line {i}" for i in range(700))}
    )
    assert decision is expected
    assert any(length > 700 and offset > 0 for length, offset in rendered)
    assert list(app.lines) == ["existing history"]
    assert screen.restored_timeout == 100
