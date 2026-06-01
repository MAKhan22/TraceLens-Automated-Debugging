#!/usr/bin/env python3
"""
main2.py — entry point for TraceLens v2 (unified fusion pipeline).

Same CLI flags as main.py; outputs go to outputs/v2/ by default.

Examples:
    python main2.py --source areeb_salem --llm --vlm
    python main2.py --trace github --llm --vlm --no-eval
"""

from v2.main import main

if __name__ == "__main__":
    main()
