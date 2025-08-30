#!/usr/bin/env python
"""LEANN CLI entry point."""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "leann-core"))

from leann_core.cli import app

if __name__ == "__main__":
    app()