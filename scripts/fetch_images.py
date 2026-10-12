#!/usr/bin/env python3
"""全 SKU の商品画像を取得する（1 req/sec）。取得済みはスキップされる。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.fetch_images import main  # noqa: E402

if __name__ == "__main__":
    main()
