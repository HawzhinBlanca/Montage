# Minimal conftest for Phase 2 final tests
import sys
from pathlib import Path

# Ensure we can import from project root
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))