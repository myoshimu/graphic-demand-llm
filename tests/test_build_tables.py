"""build_tables の不変条件。BigQuery には接続せず、出力 parquet だけを検査する。

`make build-tables` を一度も走らせていない環境では skip する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ingest.build_tables import (  # noqa: E402
    WEEK_MAX,
    WEEK_MIN,
    build_outcomes,
    derive_is_original,
)
from src.ingest.bq import load_settings  # noqa: E402

PROC = REPO / "data" / "processed"


@pytest.fixture(scope="module")
def tables():
    paths = [PROC / f for f in ("skus.parquet", "sales_weekly.parquet", "outcomes.parquet")]
    if not all(p.exists() for p in paths):
        pytest.skip("data/processed が未生成。先に `make build-tables` を実行すること")
    return tuple(pd.read_parquet(p) for p in paths)


def test_keys_unique(tables):
    skus, weekly, outcomes = tables
    assert skus["class_id"].is_unique
    assert outcomes["class_id"].is_unique
    assert not weekly.duplicated(["class_id", "week_index"]).any()


def test_referential_integrity(tables):
    skus, weekly, outcomes = tables
    ids = set(skus["class_id"])
    assert set(weekly["class_id"]) <= ids
    assert set(outcomes["class_id"]) == ids


def test_class_id_is_zero_padded(tables):
    """CSV 側で先頭ゼロが落ちた前科があるため、9 桁ゼロ埋めを明示的に守る。"""
    skus = tables[0]
    assert (skus["class_id"].str.len() == 9).all()
    assert skus["class_id"].str.isdigit().all()


def test_release_year_is_in_scope(tables):
    skus = tables[0]
    years = pd.to_datetime(skus["release_date"]).dt.year.unique()
    expected = set(load_settings()["scope"]["release_years"])
    assert set(years) == expected, f"発売年が設定と不一致: {sorted(years)} vs {sorted(expected)}"


def test_week_index_within_bounds(tables):
    weekly = tables[1]
    assert weekly["week_index"].between(WEEK_MIN, WEEK_MAX).all()


def test_preorder_is_essentially_prerelease(tables):
    """予約明細は発売前に集中しているはず。

    week 0 には僅かに残る（予約締切と発売日が同じ 7 日バケットに入るため）が、
    week 1 以降に予約が出てきたら `release_date` の定義が壊れている。
    """
    weekly = tables[1]
    assert weekly[weekly["week_index"] >= 1]["units_preorder"].sum() == 0
    pre = weekly[weekly["week_index"] < 0]["units_preorder"].sum()
    total = weekly["units_preorder"].sum()
    assert pre / total > 0.95, f"予約の発売前比率が {pre / total:.1%} しかない"


def test_launch_outcome_includes_prerelease(tables):
    """y_units_launch_8w = units_8w + 予約分。予約込み指標が壊れていないこと。"""
    outcomes = tables[2]
    expected = outcomes["units_8w"] + outcomes["units_prerelease"]
    assert (outcomes["y_units_launch_8w"] == expected).all()


def test_outcomes_match_weekly(tables):
    """outcomes が sales_weekly の単純な再集計であること（両者がずれたら片方のバグ）。"""
    skus, weekly, outcomes = tables
    recomputed = build_outcomes(weekly, skus, [4, 8])
    merged = outcomes[["class_id", "units_8w"]].merge(
        recomputed[["class_id", "units_8w"]], on="class_id", suffixes=("", "_re")
    )
    assert (merged["units_8w"] == merged["units_8w_re"]).all()


def test_returns_are_netted_not_dropped(tables):
    """返品は負数行として入っている。除外していれば units_returned が全部 0 になるはず。"""
    weekly = tables[1]
    assert weekly["units_returned"].sum() > 0


def test_n_stores_selling_is_not_used_as_denominator(tables):
    """decisions_log 2026-09-15: 配荷店舗数の代用にしない。

    `y_units_per_store_*` という列が生えていたら、分母に何を使ったかを疑う必要がある。
    A4/A5 が解決して正しい分母が入るまで、この列は存在してはいけない。
    """
    outcomes = tables[2]
    leaked = [c for c in outcomes.columns if "per_store" in c or "sellthrough" in c]
    assert not leaked, f"分母不明のまま正規化指標が作られている: {leaked}"


def test_fullprice_share_is_nan_when_no_sales(tables):
    """純販売 0 の SKU の正価比率は 0 ではなく NaN（0 埋めすると分布が歪む）。"""
    outcomes = tables[2]
    zero = outcomes[outcomes["units_8w"] <= 0]
    assert outcomes.loc[zero.index, "y_fullprice_share_8w"].isna().all()


def test_fully_observed(tables):
    """2023–2025 年発売は観測終端まで 8 週以上ある。打ち切りが混ざっていないこと。
    2026 年発売は観測途中の SKU を含む（確認では fully_observed だけを使う。decisions_log 2026-10-03（10））。"""
    skus = tables[0]
    assert skus.loc[pd.to_datetime(skus["release_date"]).dt.year <= 2025, "fully_observed"].all()


def test_is_original_fail_closed():
    """商品マスタが「オリジナル」一択かつ IP マスタが IP と言わないときだけ True。

    IP の誤判定は IP 画像の掲載につながる（§2.1）ので、曖昧なものはすべて False になることを固定する。
    """
    df = pd.DataFrame(
        {
            "item_class_names": [
                ["オリジナル"], ["オリジナル"], ["ライセンス"], ["その他"], None,
                ["オリジナル", "ライセンス"], ["オリジナル"],
            ],
            "is_ip_draft": pd.array([None, False, None, None, None, None, True], dtype="boolean"),
        }
    )
    assert derive_is_original(df).tolist() == [True, True, False, False, False, False, False]


def test_is_original_present_and_consistent(tables):
    skus, _, _ = tables
    assert skus["is_original"].notna().all()
    assert not (skus["is_original"] & (skus["item_class_name"] != "オリジナル")).any()
    # IP マスタの NULL content_id 行が結合されていないこと（2026-09-28 修正）
    assert skus.loc[skus["content_id"].isna(), "is_ip_draft"].isna().all()


def test_fill_content_from_item_master_only_maps_unique_names():
    """content_id の補完は名前が一意に対応するときだけ。曖昧なら欠損のまま残す。"""
    from src.ingest.build_tables import fill_content_from_item_master

    df = pd.DataFrame({
        "class_id": ["a", "b", "c", "d", "e", "f"],
        "content_id": ["001", None, None, "002", "003", None],
        "content_name": ["<series>", "<series>", None, "X", "X", None],
        "contents_names": [["<series>"], ["<series>"], ["<series>"], ["X"], ["X"], ["X"]],
    })
    out = fill_content_from_item_master(df).set_index("class_id")
    assert out.loc["b", "content_id"] == "001" and out.loc["b", "content_source"] == "name_map"
    assert out.loc["c", "content_name"] == "<series>" and out.loc["c", "content_id"] == "001"
    assert pd.isna(out.loc["f", "content_id"])  # X は 002/003 の 2 つに対応するので補わない
    assert out.loc["f", "content_name"] == "X"
    assert out.loc["a", "content_source"] == "orders"


def test_apply_ip_supplement_fills_only_missing(tmp_path):
    import pandas as pd
    import pytest

    from src.ingest.build_tables import apply_ip_supplement

    p = tmp_path / "s.yaml"
    p.write_text(
        "ips:\n"
        "  - {content_name: A, audience_scale: メジャー}\n"
        "  - {content_name: B, audience_scale: 判断不能}\n"
        "  - {content_name: C, audience_scale: ニッチ}\n",
        encoding="utf-8",
    )
    s = pd.DataFrame(
        {"content_name": ["A", "B", "C", "D", "A"], "ip_scale": [None, None, "中堅", None, None],
         "is_original": [False, False, False, False, True]}
    )
    out = apply_ip_supplement(s, p)
    assert pd.isna(out["ip_scale"].iloc[4])  # オリジナルには付けない
    out = out.iloc[:4]
    v = out["ip_scale"].tolist()
    assert v[0] == "メジャー" and pd.isna(v[1]) and v[2] == "中堅"  # 既存タグ（C）は上書きしない
    assert out["ip_scale_source"].tolist() == ["supplement", None, "ip_tags", None]
    p.write_text("ips:\n  - {content_name: A, audience_scale: 大}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        apply_ip_supplement(s, p)
