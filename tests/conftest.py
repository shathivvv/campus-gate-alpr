import sys
from pathlib import Path

# Let tests import the project modules (app.py, voter.py, ...) from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
