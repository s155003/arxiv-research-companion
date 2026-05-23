"""Build the index for a topic. Thin wrapper around the CLI.

Example:
    python scripts/build_index.py --query "diffusion models" --max-papers 400
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arxiv_companion.cli import build_index_cmd

if __name__ == "__main__":
    build_index_cmd()
