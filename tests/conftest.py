import importlib.util
import sys
from pathlib import Path

# Also run against an uninstalled checkout regardless of its directory name.
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "Agent_Assistant", root / "__init__.py", submodule_search_locations=[str(root)]
)
module = importlib.util.module_from_spec(spec)
sys.modules["Agent_Assistant"] = module
spec.loader.exec_module(module)
