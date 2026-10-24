"""Test configuration: expose the engine/ modules the app adds at runtime."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
