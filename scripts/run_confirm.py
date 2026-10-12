"""2 案比較の確認（2026 年発売、decisions_log 2026-10-03（10）、計画は prereg/confirm2026_plan.md）。

1. 2026 年発売の母集団のうち発売後 8 週を観測できた SKU（fully_observed）を対象にする。
2. graphic_id は人手確認をしていないので、画像ハッシュの候補クラスタ（settings.graphic_id と同じ閾値・ブロッキング）で代用する。
3. B1 / B2 を 2023–2025 年発売の全件で学習し直し、2026 年分を予測 → data/processed/confirm_baseline_predictions.parquet。
4. 2025 年と同じプロンプト・設定で 2 案比較 → data/processed/panel/pairwise_confirm2026_*.parquet。評価は scripts/eval_confirm.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import structlog
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import run_pairwise  # noqa: E402
from scripts.run_baselines import load_frame  # noqa: E402
from src.baselines import models as B  # noqa: E402
from src.features.clip_embed import embedding_config, embedding_matrix, load_cache  # noqa: E402
from src.features.graphic_id import candidate_clusters, compute_hashes  # noqa: E402

log = structlog.get_logger()
PROC = REPO / "data" / "processed"
TAG = "_confirm2026"
PRED_FILE = "confirm_baseline_predictions.parquet"
BUDGET_USD = 100.0


def auto_graphic_ids(settings: dict, years: list[int]) -> pd.DataFrame:
    """人手確認なしの graphic_id（候補クラスタ）。2023–2025 の確定済み graphic_ids.parquet には触れない。"""
    skus = pd.read_parquet(PROC / "skus.parquet")
    skus = skus[pd.to_datetime(skus["release_date"]).dt.year.isin(years)]
    man = pd.read_parquet(REPO / "data" / "cache" / "image_manifest.parquet")
    man = man[(man["status"] == "ok") & man["class_id"].isin(skus["class_id"])]
    hpath = REPO / "data" / "cache" / "image_hashes.parquet"
    hashes = pd.read_parquet(hpath) if hpath.exists() else pd.DataFrame(columns=["class_id"])
    missing = man[~man["class_id"].isin(hashes["class_id"])]
    if len(missing):
        hashes = pd.concat([hashes, compute_hashes(missing)], ignore_index=True)
        hashes.to_parquet(hpath, index=False)
    h = hashes[hashes["class_id"].isin(man["class_id"])].reset_index(drop=True)
    h = h.merge(skus[["class_id", "content_id", "content_name"]], on="class_id", how="left")
    blocks = h["content_id"].fillna("name:" + h["content_name"])
    cl = candidate_clusters(h, blocks, int(settings["graphic_id"]["threshold"]))
    cl["graphic_id"] = "auto26_" + cl["_root"].astype(str)
    return cl[["class_id", "color_id", "graphic_id"]]


def main() -> None:
    settings = yaml.safe_load((REPO / "config" / "settings.yaml").read_text(encoding="utf-8"))
    bcfg = yaml.safe_load((REPO / "config" / "baselines.yaml").read_text(encoding="utf-8"))
    models_cfg = yaml.safe_load((REPO / "config" / "models.yaml").read_text(encoding="utf-8"))
    seed = int(settings["random_seed"])
    years = settings["evaluation"]["split"]["confirm"]

    tr = load_frame(settings, bcfg).reset_index(drop=True)  # 2023–2025 年発売の全件で学習
    te = load_frame(settings, bcfg, years=years, graphic_ids=auto_graphic_ids(settings, years))
    te = te[te["fully_observed"]].reset_index(drop=True)
    cache = load_cache(REPO / embedding_config(models_cfg, "clip_vit_l14")["output"])
    data = B.Data(X_tr=tr, X_te=te, E_tr=embedding_matrix(cache, tr["image_sha256"]),
                  E_te=embedding_matrix(cache, te["image_sha256"]), y_tr=tr["y_tgt"].to_numpy(),
                  groups_tr=tr["graphic_id"].to_numpy())
    log.info("confirm_baselines", n_train=len(tr), n_confirm=len(te), n_ip=int((~te["is_original"]).sum()))
    out = te[["class_id", "color_id", "graphic_id", "is_original", "ip_scale", "y_raw", "y_tgt"]].copy()
    out["pred_B1_lgbm"] = B.b1_meta(data, bcfg, seed)["lgbm"][0]
    out["pred_B2_lgbm"] = B.b2_image(data, bcfg, seed)["lgbm"][0]
    out.to_parquet(PROC / PRED_FILE, index=False)
    run_pairwise.main(tag=TAG, pred_file=PRED_FILE, budget_usd=BUDGET_USD)


if __name__ == "__main__":
    main()
