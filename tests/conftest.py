import sys
from pathlib import Path

# ensure local package import works in test run
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
