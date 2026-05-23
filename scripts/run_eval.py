"""Run the golden-set evaluation comparing retrievers.

Examples:
    python scripts/run_eval.py
    python scripts/run_eval.py --topic diffusion
    python scripts/run_eval.py --weights "alpha=0.7,beta=0.4" --out eval/results/sweep1.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import main

if __name__ == "__main__":
    main()
