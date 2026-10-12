"""公開ベンチマーク：売上の実数が出ないこと、分位と勝ち負けが整合すること。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.benchmark.build import PUBLIC_SKU_COLUMNS, LeakError, build_pairs, build_skus, public, sales_bins

CFG = {
    "sales_bins": 10, "price_bands_jpy": [0, 3500, 4000, 5000],
    "price_band_labels": ["<3500", "3500-3999", "4000-4999", ">=5000"],
    "image_url_template": "https://example/{class_id}.{color_id}.jpg",
}


def frame() -> pd.DataFrame:
    n = 20
    return pd.DataFrame({
        "class_id": [f"{i:09d}" for i in range(n)], "color_id": ["001"] * n,
        "graphic_id": [f"g{i // 2}" for i in range(n)], "is_original": [True] * 18 + [False] * 2,
        "release_date": pd.to_datetime(["2025-01-10"] * 10 + ["2025-02-10"] * 10),
        "year": [2025] * n, "release_month": [1] * 10 + [2] * 10,
        "price_list": [2900, 3500, 3999, 4000, 4500, 5000, 5900, 3900, 4900, 3500] * 2,
        "y_raw": [float(i) for i in range(n)],
    })


def test_skus_only_original_and_no_sales_columns() -> None:
    s = build_skus(frame(), CFG, {2025: "test"})
    assert len(s) == 18
    out = public(s, PUBLIC_SKU_COLUMNS)
    assert not {"y_raw", "price_list"} & set(out.columns)
    assert out["price_band"].tolist()[:7] == ["<3500", "3500-3999", "3500-3999", "4000-4999", "4000-4999", ">=5000", ">=5000"]


def test_sales_bins_within_cohort() -> None:
    s = build_skus(frame(), CFG, {2025: "test"})
    jan = s[s.release_month == 1]
    assert jan["sales_decile"].tolist() == list(range(1, 11))  # 10 点なら順位がそのまま 1..10
    assert s.groupby("release_month")["sales_decile"].max().eq(10).all()
    y = pd.Series([1.0, 1.0, 3.0])
    assert sales_bins(y, pd.DataFrame({"m": [1, 1, 1]}), 10).tolist() == [5, 5, 10]


def test_public_rejects_sales_columns() -> None:
    with pytest.raises(LeakError):
        public(pd.DataFrame({"sku_id": ["S1"], "units_8w": [3]}), ["sku_id", "units_8w"])


def test_pairs_label_matches_sales() -> None:
    s = build_skus(frame(), CFG, {2025: "test"})
    pairs = pd.DataFrame({
        "pair_id": ["2025_0"], "set": ["original"],
        "first_class_id": ["000000001"], "first_color_id": ["001"], "first_y_raw": [1.0],
        "second_class_id": ["000000005"], "second_color_id": ["001"], "second_y_raw": [5.0],
    })
    p = build_pairs(pairs, s)
    assert p["a_sold_more"].tolist() == [0]
    a, b = (s.set_index("sku_id").loc[p[c].iloc[0], "sales_decile"] for c in ("sku_a", "sku_b"))
    assert a < b
