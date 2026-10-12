#!/usr/bin/env python3
"""画像ハッシュから graphic_id を導出する（CLAUDE.md §7-6）。make graphic-id から呼ぶ。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.graphic_id import main  # noqa: E402

if __name__ == "__main__":
    main()
