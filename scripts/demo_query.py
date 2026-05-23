"""Ask the indexed corpus a question.

Examples:
    python scripts/demo_query.py "What are the foundational papers behind diffusion models?"
    python scripts/demo_query.py --retriever vector "What is RLHF?"
    python scripts/demo_query.py --no-llm --json-out "How do transformers work?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arxiv_companion.cli import query_cmd

if __name__ == "__main__":
    query_cmd()
