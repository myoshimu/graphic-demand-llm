"""刺激条件の組み立て（CLAUDE.md §5.2）。

| 条件 | 画像 | 商品名 | 価格 | IP 名 |
|---|---|---|---|---|
| full | 商品画像 | ○ | ○ | ○ |
| image_only | 商品画像 | | | |
| artwork_only | グラフィック単体 | | | |
| text_only | | ○ | ○ | ○ |
| image_masked_text | 商品画像 | | ○ | |

商品名は pf_copy.item_master.item_name（英語、コレクション単位。IP 名を含むことが多い）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.ingest.bq import REPO_ROOT, get_client, load_settings, require, run_query

CONDITIONS = ("full", "image_only", "artwork_only", "text_only", "image_masked_text")
IMAGE_CONDITIONS = {"full", "image_only", "image_masked_text"}

ITEM_NAME_SQL = """
SELECT class_id, color_id, ANY_VALUE(item_name) AS item_name, ANY_VALUE(color_name) AS color_name
FROM `{item_master}`
GROUP BY class_id, color_id
"""


class Stimulus(BaseModel):
    class_id: str
    color_id: str
    condition: str
    text: str
    image_path: str | None
    image_sha256: str | None
    is_original: bool


def load_item_names(refresh: bool = False) -> pd.DataFrame:
    """商品名を BQ から取得し data/processed/item_names.parquet にキャッシュする（約 0.01 GB）。"""
    s = load_settings()
    path = REPO_ROOT / require(s, "paths.processed") / "item_names.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = run_query(get_client(s), ITEM_NAME_SQL.format(item_master=require(s, "bigquery.item_master")), label="item_names")
    df.to_parquet(path, index=False)
    return df


def load_sku_frame() -> pd.DataFrame:
    """パネルに必要な列を SKU 単位で結合する（母集団に限らない。絞り込みは呼び出し側）。"""
    s = load_settings()
    proc = REPO_ROOT / require(s, "paths.processed")
    skus = pd.read_parquet(proc / "skus.parquet")[
        ["class_id", "color_id", "release_date", "ip_name", "is_original", "ip_scale", "price_list"]
    ]
    man = pd.read_parquet(REPO_ROOT / "data" / "cache" / "image_manifest.parquet")
    man = man.loc[man["status"] == "ok", ["class_id", "color_id", "image_path", "image_sha256"]]
    df = skus.merge(load_item_names(), on=["class_id", "color_id"], how="left").merge(
        man, on=["class_id", "color_id"], how="left"
    )
    return df


def _ip_line(row: pd.Series) -> str | None:
    if pd.isna(row["ip_name"]):
        return "retailer オリジナルデザイン" if row["is_original"] else None
    kind = "retailer オリジナルのシリーズ" if row["is_original"] else "コラボ（ライセンス）作品"
    return f"{kind}：{row['ip_name']}"


def build_stimulus(row: pd.Series, condition: str) -> Stimulus:
    if condition not in CONDITIONS:
        raise ValueError(f"未知の条件: {condition}")
    if condition == "artwork_only":
        # グラフィック単体画像は §5.4 の切り抜き実装後に対応する
        raise NotImplementedError("artwork_only は crop_artwork（§5.4）の実装後に有効化する")
    price = f"価格：{int(row['price_list']):,} 円（税込）"
    lines: list[str] = []
    if condition in ("full", "text_only"):
        if pd.notna(row.get("item_name")):
            lines.append(f"商品名：{row['item_name']}")
        if (ip := _ip_line(row)) is not None:
            lines.append(ip)
        lines.append(price)
    elif condition == "image_masked_text":
        lines.append(price)
    if condition in IMAGE_CONDITIONS:
        lines.insert(0, "画像の T シャツです。")
    use_image = condition in IMAGE_CONDITIONS
    if use_image and pd.isna(row.get("image_path")):
        raise ValueError(f"{row['class_id']}.{row['color_id']} に画像がありません")
    return Stimulus(
        class_id=row["class_id"],
        color_id=row["color_id"],
        condition=condition,
        text="\n".join(lines),
        image_path=str(Path(row["image_path"])) if use_image else None,
        image_sha256=row["image_sha256"] if use_image else None,
        is_original=bool(row["is_original"]),
    )
