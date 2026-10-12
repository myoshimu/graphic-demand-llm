#!/usr/bin/env python3
"""make build-tables のエントリポイント。src/ingest/build_tables.py を呼ぶだけ。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.build_tables import main  # noqa: E402

if __name__ == "__main__":
    main()
