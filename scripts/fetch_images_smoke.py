#!/usr/bin/env python3
"""§7-4 のスモークテスト。まず 20 件だけ取得して動作確認する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.bq import load_settings, require  # noqa: E402
from src.ingest.fetch_images import main  # noqa: E402

if __name__ == "__main__":
    main(limit=int(require(load_settings(), "images.smoke_test_n")))
