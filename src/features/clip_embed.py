"""商品画像の open_clip 埋め込み（ベースライン B2 / B4、CLAUDE.md §5.1）。

キャッシュは image_sha256 をキーにした parquet。計算済みの画像は再計算しない。
埋め込みは L2 正規化済み（コサイン類似度 = 内積）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
from PIL import Image

log = structlog.get_logger()


def embedding_config(models_cfg: dict, key: str) -> dict:
    for e in models_cfg["embeddings"]:
        if e["key"] == key:
            if not e.get("enabled") or not e.get("pretrained"):
                raise ValueError(f"埋め込み {key} が未確定（config/models.yaml）")
            return e
    raise KeyError(key)


def load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({"image_sha256": pd.Series(dtype="str"), "embedding": pd.Series(dtype="object")})


def embed_images(manifest: pd.DataFrame, cfg: dict, repo: Path) -> pd.DataFrame:
    """manifest（image_path, image_sha256）の未計算分だけ埋め込んでキャッシュに追記し、全体を返す。"""
    import open_clip

    out_path = repo / cfg["output"]
    cache = load_cache(out_path)
    todo = manifest[~manifest["image_sha256"].isin(cache["image_sha256"])].drop_duplicates("image_sha256")
    log.info("clip_embed", model=cfg["model_id"], pretrained=cfg["pretrained"], cached=len(cache), todo=len(todo))
    if todo.empty:
        return cache

    torch.manual_seed(0)
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["model_id"], pretrained=cfg["pretrained"])
    model.eval()
    bs = int(cfg["batch_size"])
    rows: list[dict] = []
    for i in range(0, len(todo), bs):
        chunk = todo.iloc[i : i + bs]
        x = torch.stack([preprocess(Image.open(repo / p).convert("RGB")) for p in chunk["image_path"]])
        with torch.no_grad():
            z = model.encode_image(x)
        z = torch.nn.functional.normalize(z, dim=-1).numpy().astype(np.float32)
        rows += [{"image_sha256": h, "embedding": v} for h, v in zip(chunk["image_sha256"], z, strict=True)]
        # 途中で落ちても続きから再開できるよう定期的に書き出す
        if len(rows) >= 20 * bs or i + bs >= len(todo):
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(out_path, index=False)
            log.info("clip_embed_progress", done=len(cache))
            rows = []
    return cache


def embedding_matrix(cache: pd.DataFrame, shas: pd.Series) -> np.ndarray:
    lut = dict(zip(cache["image_sha256"], cache["embedding"], strict=True))
    missing = [h for h in shas if h not in lut]
    if missing:
        raise KeyError(f"埋め込みが無い画像 {len(missing)} 件（make embeddings を先に実行）")
    return np.stack([np.asarray(lut[h], dtype=np.float32) for h in shas])
