"""段階 1（見極め）：テスト期間（2025 年発売）の母集団に Pro のパネルをかける（decisions_log 2026-10-03（6））。

条件 full / image_masked_text × ペルソナ 50、温度 0、seed 1、言い換え 0、画像 high、オンライン単価で上限 $260。
出力は data/processed/panel/stage1_responses.parquet と集計 stage1_scores.parquet。評価は scripts/eval_panel_stage1.py。
キャッシュがあるので、途中で止まっても再実行すれば続きから回る。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import structlog
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_baselines import load_frame  # noqa: E402
from src.ingest.bq import REPO_ROOT, load_settings, require  # noqa: E402
from src.panel.llm import GeminiPanelClient  # noqa: E402
from src.panel.personas import sample_personas  # noqa: E402
from src.panel.run import aggregate, run_panel  # noqa: E402
from src.panel.stimuli import load_sku_frame  # noqa: E402

log = structlog.get_logger()

MODEL_KEY = "api_a"
CONDITIONS = ["full", "image_masked_text"]
BUDGET_USD = 260.0


def test_skus(settings: dict) -> pd.DataFrame:
    bcfg = yaml.safe_load((REPO_ROOT / "config" / "baselines.yaml").read_text(encoding="utf-8"))
    te = load_frame(settings, bcfg).query("split == 'test'")[["class_id", "color_id"]]
    return load_sku_frame().merge(te, on=["class_id", "color_id"])


def main() -> None:
    settings = load_settings()
    skus = test_skus(settings)
    out_dir = REPO_ROOT / require(settings, "paths.processed") / "panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiPanelClient(MODEL_KEY, budget_usd=BUDGET_USD, media_resolution="high")
    resp = run_panel(client, skus, CONDITIONS, sample_personas(50), temperature=0.0, seed=1, workers=16)
    resp.to_parquet(out_dir / "stage1_responses.parquet", index=False)
    aggregate(resp).to_parquet(out_dir / "stage1_scores.parquet", index=False)
    log.info("stage1_done", n_skus=len(skus), n_responses=len(resp), spent_usd=round(client.spent_usd, 2))


if __name__ == "__main__":
    main()
