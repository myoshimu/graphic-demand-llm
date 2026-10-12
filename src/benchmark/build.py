"""公開ベンチマークの生成（CLAUDE.md §5 フェーズ3、config/benchmark.yaml、decisions_log 2026-10-04（17）（18））。

retailer オリジナルのみ。売上は「同じ発売月のオリジナル内での 8 週 EC 販売数の分位」と「組の勝ち負け」だけを出す。
販売数・金額・週次推移・在庫・生産数は出さない（PUBLIC_*_COLUMNS 以外の列は書き出さない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PUBLIC_SKU_COLUMNS = [
    "sku_id", "class_id", "color_id", "design_id", "release_year", "release_month", "split",
    "price_band", "sales_decile", "cohort_size", "image_url",
]
PUBLIC_PAIR_COLUMNS = ["pair_id", "split", "sku_a", "sku_b", "a_sold_more"]


class LeakError(ValueError):
    pass


def price_band(price: pd.Series, edges: list[int], labels: list[str]) -> pd.Series:
    idx = np.searchsorted(np.asarray(edges), price.to_numpy(float), side="right") - 1
    return pd.Series(np.asarray(labels)[idx], index=price.index)


def sales_bins(y: pd.Series, cohort: pd.DataFrame, n_bins: int) -> pd.Series:
    """コホート内の順位を 1..n_bins に。同順位は平均順位。上位ほど大きい値。"""
    pct = y.groupby([cohort[c] for c in cohort.columns]).rank(method="average", pct=True)
    return np.ceil(pct * n_bins).clip(1, n_bins).astype(int)


def build_skus(frame: pd.DataFrame, cfg: dict, split_of_year: dict[int, str]) -> pd.DataFrame:
    d = frame[frame["is_original"].astype(bool)].copy()
    d = d.sort_values(["release_date", "class_id", "color_id"]).reset_index(drop=True)
    d["release_year"] = d["year"].astype(int)
    d["split"] = d["release_year"].map(split_of_year)
    if d["split"].isna().any():
        raise ValueError("split の決まらない年がある")
    d["sku_id"] = [f"S{i:05d}" for i in range(len(d))]
    codes = {g: f"D{i:05d}" for i, g in enumerate(pd.unique(d["graphic_id"]))}
    d["design_id"] = d["graphic_id"].map(codes)
    d["price_band"] = price_band(d["price_list"], cfg["price_bands_jpy"], cfg["price_band_labels"])
    coh = d[["year", "release_month"]]
    d["sales_decile"] = sales_bins(d["y_raw"], coh, int(cfg["sales_bins"]))
    d["cohort_size"] = d.groupby(["year", "release_month"])["sku_id"].transform("size")
    d["image_url"] = [cfg["image_url_template"].format(class_id=c, color_id=k) for c, k in zip(d.class_id, d.color_id)]
    return d


def build_pairs(pairs: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """評価に使ったオリジナルの組（2 案比較）。a_sold_more は sku_a の 8 週 EC 販売数が多ければ 1。"""
    key = skus.set_index(["class_id", "color_id"])["sku_id"]
    p = pairs[pairs["set"] == "original"].copy()
    p["sku_a"] = [key.get((c, k)) for c, k in zip(p.first_class_id, p.first_color_id)]
    p["sku_b"] = [key.get((c, k)) for c, k in zip(p.second_class_id, p.second_color_id)]
    if p[["sku_a", "sku_b"]].isna().any().any():
        raise ValueError("組に母集団外の SKU がある")
    p["a_sold_more"] = (p.first_y_raw > p.second_y_raw).astype(int)
    p["split"] = p["sku_a"].map(skus.set_index("sku_id")["split"])
    return p


def public(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """許可した列だけを返す。実数の売上に当たる列名が紛れていたら止める。"""
    out = df[cols].copy()
    banned = ("units", "revenue", "y_raw", "y_tgt", "stock", "production", "price_list")
    bad = [c for c in out.columns if any(b in c for b in banned)]
    if bad:
        raise LeakError(f"公開できない列: {bad}")
    return out
