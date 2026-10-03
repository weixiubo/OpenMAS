"""Test-only adapters for byte-frozen historical workflows against today's library.

The historical check separately runs these workflows with their original source.
No compatibility module is installed with either public wheel.
"""

from __future__ import annotations

import atexit
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

from openagora_experiments.egor import parsing, prompts, task_model

_temporary = TemporaryDirectory(prefix="openmas-workflow-tests-")
atexit.register(_temporary.cleanup)
_root = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("openmas_frozen_checker", _root / "scripts/check_historical.py")
assert _spec and _spec.loader
_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checker)
_destination = Path(_temporary.name)
_checker.extract_verified(_root, _destination)

_support = ModuleType("openmas_test_history")
_support.historical_root = lambda: _destination
sys.modules[_support.__name__] = _support
sys.path.insert(0, str(_destination / "code/experiments/scripts"))

# Only old import locations change. Function objects and assertions use current code.
_protocol = ModuleType("openagora_experiments.egor.protocol")
for _module in (task_model, prompts, parsing):
    for _name, _value in vars(_module).items():
        if not _name.startswith("__"):
            setattr(_protocol, _name, _value)
sys.modules[_protocol.__name__] = _protocol
