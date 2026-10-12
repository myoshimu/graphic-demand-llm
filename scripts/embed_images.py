#!/usr/bin/env python3
"""分析母集団の商品画像を open_clip で埋め込む（B2 / B4 の入力）。引数なしで再実行可能。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.features.clip_embed import embed_images, embedding_config  # noqa: E402


def main() -> None:
    models = yaml.safe_load((REPO / "config" / "models.yaml").read_text(encoding="utf-8"))
    cfg = embedding_config(models, "clip_vit_l14")
    man = pd.read_parquet(REPO / "data" / "cache" / "image_manifest.parquet")
    pop = pd.read_parquet(REPO / "data" / "processed" / "population.parquet")
    man = man[man["status"] == "ok"].merge(
        pop.loc[pop["in_population"].fillna(False).astype(bool), ["class_id", "color_id"]], on=["class_id", "color_id"]
    )
    cache = embed_images(man, cfg, REPO)
    print(f"embeddings: {man['image_sha256'].isin(cache['image_sha256']).sum()}/{len(man)}")


if __name__ == "__main__":
    main()
