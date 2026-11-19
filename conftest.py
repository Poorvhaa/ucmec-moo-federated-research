import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
