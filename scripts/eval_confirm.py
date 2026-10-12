"""2 案比較の確認（2026 年発売）の評価 → reports/pairwise_confirm2026.md。合格基準は 2025 年と同じ（decisions_log 2026-10-03（10））。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_pairwise import main  # noqa: E402

if __name__ == "__main__":
    main(tag="_confirm2026", period="確認用データ（2026 年 1〜6 月発売、発売後 8 週を観測できた SKU）")
