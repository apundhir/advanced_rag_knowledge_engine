from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable as a package (so `import app` works in tests)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
