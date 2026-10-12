#!/usr/bin/env python3
"""分析母集団（画像あり かつ 発売時 EC 掲載）を確定する（docs/questions.md A31）。make population から呼ぶ。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.ec_listed import build_population  # noqa: E402

if __name__ == "__main__":
    build_population()
