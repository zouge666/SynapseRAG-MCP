import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ["SYNAPSERAG_PUBLIC"] = "1"

from observability.dashboard.app import main


main()
