#!/usr/bin/env python3
"""閲覧ログから EC 掲載フラグを導出する（docs/questions.md A19）。make ec-listed から呼ぶ。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.ec_listed import main  # noqa: E402

if __name__ == "__main__":
    main()
