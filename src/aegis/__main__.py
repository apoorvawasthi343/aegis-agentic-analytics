"""AEGIS CLI - command-line interface for the AEGIS framework.

Provides python -m aegis to run the full pipeline from the terminal.
"""

from src.aegis.main import build_parser, main, run_cli

if __name__ == "__main__":
    main()
