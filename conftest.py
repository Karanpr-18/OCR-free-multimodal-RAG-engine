"""
Root conftest.py — ensures the project root is on sys.path so that
`import src.*` works in pytest without needing `pip install -e .`.
"""
import sys
from pathlib import Path

# Insert project root so `src.*` imports resolve in-tree
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
