import sys
from pathlib import Path

# Ensure the repository root (fin101/fin101) is on sys.path so tests can import `src`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
