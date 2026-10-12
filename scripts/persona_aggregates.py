"""ペルソナ設計用の顧客分布を BigQuery 上で集計する（CLAUDE.md §5.2、questions.md B3）。

対象顧客 = 分析期間（settings.scope.release_years）に EC で主カテゴリ（U_Tシャツ）を購入した顧客。
**個票は取得しない**（§2.2）。BQ 側で顧客数に集計した表だけを受け取り、
min_cell 未満のセルは伏せる。出力は data/processed/persona_aggregates.parquet と
reports/persona_distribution.md。

水準の閾値（購買頻度・IP ファン度の区切り）はここでは決めない（§2.5）。分布だけを出す。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest.bq import get_client, load_settings, require, run_query  # noqa: E402

logger = logging.getLogger(__name__)

MIN_CELL = 10  # 再識別防止のため、これ未満の顧客数は伏せる

# 年齢は分析期間の中点で計算する。config/personas.yaml の age_band と同じ区切り
SQL = """
WITH orig AS (
  -- IP／オリジナル区分は orders に無く、商品マスタの item_class_name にある。
  -- build_tables.derive_is_original と同じく「オリジナル」一択のみ True、マスタに無い class_id は IP 側
  SELECT class_id, LOGICAL_AND(item_class_name = @original_label) AS is_original
  FROM `{item_master}`
  GROUP BY class_id
),
buyers AS (
  SELECT
    customer_id,
    COUNT(DISTINCT order_no)                                        AS n_orders,
    SUM(quantity)                                                   AS units,
    SUM(IF(IFNULL(orig.is_original, FALSE), 0, quantity))           AS units_ip
  FROM `{orders}` LEFT JOIN orig USING (class_id)
  WHERE item_class_name_2 = @category
    AND store_category = 'EC'
    AND cancel_flg = 0
    AND quantity > 0
    AND customer_id IS NOT NULL
    AND DATE(order_date) BETWEEN @date_from AND @date_to
    AND hanbai_kbn NOT IN UNNEST(@exclude_hanbai)
    AND production_kbn NOT IN UNNEST(@exclude_production)
  GROUP BY customer_id
),
joined AS (
  SELECT
    b.n_orders,
    b.units,
    SAFE_DIVIDE(b.units_ip, b.units) AS ip_share,
    NULLIF(TRIM(c.sex), '')         AS sex,
    NULLIF(TRIM(c.prefectures), '') AS prefectures,
    DATE_DIFF(@age_ref, DATE(c.birthday), YEAR) AS age
  FROM buyers b
  LEFT JOIN `{customers}` c USING (customer_id)
)
SELECT
  CASE
    WHEN age IS NULL OR age < 10 OR age > 90 THEN 'unknown'
    WHEN age < 20 THEN '-19' WHEN age < 25 THEN '20-24' WHEN age < 30 THEN '25-29'
    WHEN age < 35 THEN '30-34' WHEN age < 40 THEN '35-39' WHEN age < 50 THEN '40-49'
    ELSE '50-' END                                  AS age_band,
  IFNULL(sex, 'unknown')                             AS sex,
  IFNULL(prefectures, 'unknown')                     AS prefecture,
  LEAST(n_orders, 10)                                AS n_orders_capped,
  CAST(FLOOR(IFNULL(ip_share, 0) * 4) AS INT64)      AS ip_share_q,  -- 0:<25% 1:<50% 2:<75% 3:<100% 4:=100%
  COUNT(*)                                           AS n_customers,
  SUM(units)                                         AS units
FROM joined
GROUP BY 1, 2, 3, 4, 5
"""


def _marginal(df: pd.DataFrame, col: str) -> pd.DataFrame:
    m = df.groupby(col, dropna=False)[["n_customers", "units"]].sum()
    m["share_customers"] = m["n_customers"] / m["n_customers"].sum()
    m["share_units"] = m["units"] / m["units"].sum()
    m.loc[m["n_customers"] < MIN_CELL, ["n_customers", "units"]] = pd.NA
    return m.reset_index()


def _md_table(m: pd.DataFrame, col: str) -> str:
    lines = [f"| {col} | 顧客数 | 顧客比 | 購入点数比 |", "|---|---:|---:|---:|"]
    for r in m.itertuples(index=False):
        n = "<10" if pd.isna(r.n_customers) else f"{int(r.n_customers):,}"
        lines.append(f"| {getattr(r, col)} | {n} | {r.share_customers:.1%} | {r.share_units:.1%} |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = load_settings()
    years = require(s, "scope.release_years")
    date_from, date_to = f"{min(years)}-01-01", f"{max(years)}-12-31"
    mid_year = (min(years) + max(years)) // 2
    params = [
        bigquery.ScalarQueryParameter("category", "STRING", require(s, "scope.primary_category")),
        bigquery.ScalarQueryParameter("original_label", "STRING", "オリジナル"),
        bigquery.ScalarQueryParameter("date_from", "DATE", date_from),
        bigquery.ScalarQueryParameter("date_to", "DATE", date_to),
        bigquery.ScalarQueryParameter("age_ref", "DATE", f"{mid_year}-07-01"),
        bigquery.ArrayQueryParameter("exclude_hanbai", "STRING", require(s, "scope.exclude_hanbai_kbn")),
        bigquery.ArrayQueryParameter("exclude_production", "STRING", require(s, "scope.exclude_production_kbn")),
    ]
    sql = SQL.format(orders=require(s, "bigquery.orders"), customers=require(s, "bigquery.customers"),
                     item_master=require(s, "bigquery.item_master"))
    df = run_query(get_client(s), sql, params, label="persona_aggregates", max_scan_gb=20.0)

    out = REPO_ROOT / require(s, "paths.processed") / "persona_aggregates.parquet"
    # 細かいセル（5 軸の組合せ）は保存しない。周辺分布だけを残す
    marg = {c: _marginal(df, c) for c in ["age_band", "sex", "prefecture", "n_orders_capped", "ip_share_q"]}
    pd.concat([m.rename(columns={c: "level"}).assign(axis=c) for c, m in marg.items()]).astype(
        {"level": str}
    ).to_parquet(out, index=False)

    total = int(df["n_customers"].sum())
    ip_labels = {0: "IP 比率 <25%", 1: "25–50%", 2: "50–75%", 3: "75–100% 未満", 4: "100%（IP のみ）"}
    marg["ip_share_q"]["ip_share_q"] = marg["ip_share_q"]["ip_share_q"].map(ip_labels)
    marg["n_orders_capped"]["n_orders_capped"] = marg["n_orders_capped"]["n_orders_capped"].map(
        lambda v: f"{v} 回" if v < 10 else "10 回以上"
    )
    pref = marg["prefecture"].sort_values("n_customers", ascending=False, na_position="last")

    body = [
        "# ペルソナ設計用の顧客分布（questions.md B3）",
        "",
        f"> 対象：{date_from}〜{date_to} に EC で {require(s, 'scope.primary_category')} を購入した顧客 "
        f"**{total:,} 人**（キャンセル除外、受注生産除外）。BQ 上で集計し、個票は取得していない。",
        f"> 年齢は {mid_year}-07-01 時点。顧客数 {MIN_CELL} 未満のセルは伏せた。",
        "> 「購入点数比」は購入点数で重み付けした比率（需要側の分布）。どちらに合わせるかは B3 で決める。",
        "",
        "## 年齢", "", _md_table(marg["age_band"], "age_band"), "",
        "## 性別", "", _md_table(marg["sex"], "sex"), "",
        "## 期間中の T シャツ EC 注文回数（購買頻度の元）", "", _md_table(marg["n_orders_capped"], "n_orders_capped"), "",
        "## T シャツ購入点数に占める IP 商品の比率（IP ファン度の代理）", "", _md_table(marg["ip_share_q"], "ip_share_q"), "",
        "## 都道府県（上位 15）", "", _md_table(pref.head(15), "prefecture"), "",
    ]
    rep = REPO_ROOT / require(s, "paths.reports") / "persona_distribution.md"
    rep.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {rep} ({total:,} customers)")


if __name__ == "__main__":
    main()
