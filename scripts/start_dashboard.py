from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit("streamlit is not installed")
    root = Path(__file__).resolve().parents[1]
    app_path = root / "src" / "observability" / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
