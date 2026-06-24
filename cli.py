from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

main = cast(Callable[[], None], import_module("Agent_Assistant.cli_app.main").main)


if __name__ == "__main__":
    main()
