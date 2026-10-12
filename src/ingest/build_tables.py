"""wsdm.orders から skus / sales_weekly / outcomes の 3 テーブルを materialize する。

CLAUDE.md §3.3 の正規化テーブルのうち、`graphics`（graphic_id 単位）は pHash
クラスタリング（§7-6）が済むまで作れないため、ここでは作らない。

粒度の定義（docs/data_dictionary.md §0 と対応）:
  - class_id = デザイン × ボディ色。1 class_id ≒ 1 色 × 約 4 サイズ。画像 URL はこの単位。
  - skus         : class_id 単位（1 行 = 1 商品色）
  - sales_weekly : class_id × 発売からの週（week_index）
  - outcomes     : class_id 単位の集計アウトカム（§3.4 のうち分母不要のもののみ）

重要な設計判断（docs/decisions_log.md 参照）:
  - 返品は voucher_type='店舗売上返品伝票' の負数行として入っているので、SUM(quantity)
    がそのまま純販売数になる。行を除外してはいけない。
  - release_date は hanbai_kbn='通常' の初回販売日。予約（hanbai_kbn='予約'）は発売前に
    order_date が立つため、これを起点にすると SKU 間で基準日がずれる。予約分は
    week_index が負の行として残し、units_preorder に別集計する。
  - 受注生産（hanbai_kbn / production_kbn）は questions.md A9 に従い全除外。
  - n_stores_selling は「1 点以上売れた店舗数」であって配荷店舗数ではない。売れない商品ほど
    下方バイアスがかかるので、y_units_per_store_* の分母には決して使わない
    （docs/decisions_log.md 2026-09-15）。配荷店舗数は A5 / E9 の解決待ち。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import pandas as pd
from google.cloud import bigquery
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .bq import REPO_ROOT, get_client, load_settings, require, run_query

logger = logging.getLogger(__name__)

# 発売前 12 週（予約）〜発売後 104 週に限定して週次パネルを切る。
WEEK_MIN, WEEK_MAX = -12, 103

PANEL_SQL = """
WITH lines AS (
  SELECT
    class_id,
    color_id,
    size_id,
    DATE(order_date) AS sale_date,
    quantity,
    price,
    total_price,
    sale_kbn,
    hanbai_kbn,
    store_id,
    store_category,
    content_id,
    content_name,
    creator_name,
    design_category_name,
    season_name,
    item_class_name_1,
    item_class_name_3,
    item_category_name
  FROM `{orders}`
  WHERE item_class_name_2 = @category
    AND cancel_flg = 0
    AND hanbai_kbn NOT IN UNNEST(@exclude_hanbai)
    AND production_kbn NOT IN UNNEST(@exclude_production)
),
release AS (
  -- 発売日 = 通常販売の初回実売日（予約・返品を起点にしない）
  SELECT
    class_id,
    MIN(IF(hanbai_kbn = '通常' AND quantity > 0, sale_date, NULL)) AS release_date,
    MIN(IF(quantity > 0, sale_date, NULL))                         AS first_transaction_date
  FROM lines
  GROUP BY class_id
),
cohort AS (
  SELECT class_id, release_date, first_transaction_date
  FROM release
  WHERE release_date >= @period_start AND release_date < @period_end
),
attrs AS (
  SELECT
    l.class_id,
    COUNT(DISTINCT l.color_id)            AS n_color_ids,
    ANY_VALUE(l.color_id)                 AS color_id,
    COUNT(DISTINCT l.size_id)             AS n_sizes,
    COUNT(DISTINCT l.content_id)          AS n_content_ids,
    MAX(l.content_id)                     AS content_id,
    MAX(l.content_name)                   AS content_name,
    MAX(l.creator_name)                   AS creator_name,
    MAX(l.design_category_name)           AS design_category_name,
    MAX(l.season_name)                    AS season_name,
    MAX(l.item_class_name_1)              AS item_class_name_1,
    MAX(l.item_class_name_3)              AS item_class_name_3,
    MAX(l.item_category_name)             AS item_category_name,
    MAX(IF(l.sale_kbn = 'プロパー', l.price, NULL)) AS price_proper_max,
    MIN(IF(l.sale_kbn = 'プロパー', l.price, NULL)) AS price_proper_min,
    SUM(l.quantity)                       AS units_lifetime,
    SUM(l.total_price)                    AS revenue_lifetime,
    MAX(l.sale_date)                      AS last_sale_date,
    LOGICAL_OR(l.hanbai_kbn = '予約')     AS has_preorder
  FROM lines l
  JOIN cohort c USING (class_id)
  GROUP BY l.class_id
),
weekly AS (
  SELECT
    l.class_id,
    -- FLOOR 除算。発売前（予約）は負の week_index になる
    CAST(FLOOR(DATE_DIFF(l.sale_date, c.release_date, DAY) / 7) AS INT64) AS week_index,
    SUM(l.quantity)                                                    AS units,
    SUM(IF(l.sale_kbn = 'プロパー', l.quantity, 0))                    AS units_fullprice,
    SUM(IF(l.sale_kbn = 'セール', l.quantity, 0))                      AS units_sale,
    SUM(IF(l.store_category = 'EC', l.quantity, 0))                    AS units_ec,
    SUM(IF(l.store_category = 'プロパー', l.quantity, 0))              AS units_store_proper,
    SUM(IF(l.store_category = 'アウトレット', l.quantity, 0))          AS units_outlet,
    SUM(IF(l.hanbai_kbn = '予約', l.quantity, 0))                      AS units_preorder,
    SUM(IF(l.quantity < 0, -l.quantity, 0))                            AS units_returned,
    SUM(l.total_price)                                                 AS revenue,
    COUNT(DISTINCT IF(l.quantity > 0, l.store_id, NULL))               AS n_stores_selling
  FROM lines l
  JOIN cohort c USING (class_id)
  GROUP BY class_id, week_index
)
SELECT
  w.class_id,
  w.week_index,
  c.release_date,
  c.first_transaction_date,
  w.units,
  w.units_fullprice,
  w.units_sale,
  w.units_ec,
  w.units_store_proper,
  w.units_outlet,
  w.units_preorder,
  w.units_returned,
  w.revenue,
  w.n_stores_selling,
  a.color_id,
  a.n_color_ids,
  a.n_sizes,
  a.content_id,
  a.n_content_ids,
  a.content_name,
  a.creator_name,
  a.design_category_name,
  a.season_name,
  a.item_class_name_1,
  a.item_class_name_3,
  a.item_category_name,
  a.price_proper_max,
  a.price_proper_min,
  a.units_lifetime,
  a.revenue_lifetime,
  a.last_sale_date,
  a.has_preorder
FROM weekly w
JOIN cohort c USING (class_id)
JOIN attrs  a USING (class_id)
WHERE w.week_index BETWEEN @week_min AND @week_max
ORDER BY w.class_id, w.week_index
"""

# content_id -> IP 属性。小さいテーブルなので別クエリで引く。
# collection_ip_master は (collection_slug, content_id) 粒度なので content_id に畳む。
IP_SQL = """
WITH m AS (
  SELECT
    content_id,
    MAX(ip_name_canonical)  AS ip_name,
    LOGICAL_OR(CAST(is_ip_draft AS BOOL)) AS is_ip_draft,
    LOGICAL_OR(CAST(resolved AS BOOL))    AS ip_resolved,
    SUM(users)              AS ip_users,
    SUM(paid_users)         AS ip_paid_users,
    SUM(pageviews)          AS ip_pageviews
  FROM `{ip_master}`
  GROUP BY content_id
),
t AS (
  SELECT
    content_id,
    MAX(audience_scale)    AS ip_scale,
    MAX(origin)            AS ip_origin,
    MAX(medium)            AS ip_medium,
    MAX(era)               AS ip_era,
    MAX(fandom_intensity)  AS ip_fandom_intensity,
    MAX(gender_skew)       AS ip_gender_skew
  FROM `{ip_tags}`
  GROUP BY content_id
)
SELECT
  m.content_id, m.ip_name, m.is_ip_draft, m.ip_resolved,
  m.ip_users, m.ip_paid_users, m.ip_pageviews,
  t.ip_scale, t.ip_origin, t.ip_medium, t.ip_era,
  t.ip_fandom_intensity, t.ip_gender_skew
FROM m LEFT JOIN t USING (content_id)
"""

# class_id -> 商品マスタのオリジナル／ライセンス区分（questions.md A7）。
# item_master は item_code（サイズ）粒度なので class_id に畳む。区分は class_id 内で一意（実測）。
ITEM_CLASS_SQL = """
SELECT
  class_id,
  ARRAY_AGG(DISTINCT item_class_name IGNORE NULLS) AS item_class_names,
  ARRAY_AGG(DISTINCT contents_name IGNORE NULLS)   AS contents_names,
  -- 正価（税込）は商品マスタの値を使う。受注明細の MAX(price) は販売行が多いほど大きくなり
  -- アウトカムが漏れる（2026-09-29 発見、decisions_log）。マスタは取得時点のスナップショット
  MIN(sales_price_tax_included)                    AS price_list,
  COUNT(DISTINCT sales_price_tax_included)         AS n_list_prices
FROM `{item_master}`
GROUP BY class_id
"""

ORIGINAL_LABEL = "オリジナル"
IP_SCALE_LEVELS = ("ニッチ", "中堅", "メジャー", "グローバルメガ")


def apply_ip_supplement(skus: pd.DataFrame, path: Path) -> pd.DataFrame:
    """wsdm.ip_tags に行が無い IP の ip_scale を config/ip_tags_supplement.yaml（研究責任者の手作業）で補う。

    対象はライセンス IP（is_original == False）のみ。ip_tags 側に値がある SKU は上書きしない。null と「判断不能」は欠損のまま残す（questions.md A32）。
    """
    out = skus.copy()
    out["ip_scale_source"] = pd.Series(np.where(out["ip_scale"].notna(), "ip_tags", None), index=out.index, dtype="object")
    if not path.exists():
        return out
    rows = yaml.safe_load(path.read_text(encoding="utf-8")).get("ips") or []
    tags: dict[str, str] = {}
    for r in rows:
        v = r.get("audience_scale")
        if v in (None, "判断不能", "不明"):  # どちらも未判定として欠損のまま
            continue
        if v not in IP_SCALE_LEVELS:
            raise ValueError(f"{path.name}: {r['content_name']!r} の audience_scale が不正: {v!r}")
        tags[r["content_name"]] = v
    fill = out["ip_scale"].isna() & out["content_name"].isin(tags) & ~out["is_original"].astype(bool)
    out.loc[fill, "ip_scale"] = out.loc[fill, "content_name"].map(tags)
    out.loc[fill, "ip_scale_source"] = "supplement"
    logger.info("ip_supplement filled=%d tagged_ips=%d", int(fill.sum()), len(tags))
    return out


def fill_content_from_item_master(skus: pd.DataFrame) -> pd.DataFrame:
    """受注明細に content_id が無い SKU の content_id / content_name を補う。

    2025 年発売分は受注明細の content_id が 177/635 点で欠けていた（2023–24 はほぼ 0）。
    名前（`content_name`）は受注明細にあることが多く、無ければ商品マスタの `contents_name`
    （同じ値、questions.md A30）で補う。content_id は、同じ名前に content_id が 1 つだけ
    対応する場合に限り補う。由来は `content_source`（orders / name_map / 欠損）に残す。
    """
    out = skus.copy()
    names = out["contents_names"].map(lambda v: v[0] if v is not None and len(v) == 1 else None)
    missing = out["content_name"].isna() & names.notna()
    out.loc[missing, "content_name"] = names[missing]

    out["content_source"] = pd.Series(
        np.where(out["content_id"].notna(), "orders", None), index=out.index, dtype="object")
    known = out.loc[out["content_id"].notna(), ["content_name", "content_id"]].drop_duplicates()
    unique = known.groupby("content_name")["content_id"].agg(lambda x: x.iloc[0] if len(x) == 1 else None)
    need = out["content_id"].isna() & out["content_name"].notna()
    out.loc[need, "content_id"] = out.loc[need, "content_name"].map(unique)
    out.loc[need & out["content_id"].notna(), "content_source"] = "name_map"
    return out.drop(columns="contents_names")


def derive_is_original(skus: pd.DataFrame) -> pd.Series:
    """`is_original` を fail-closed で決める。

    IP を誤ってオリジナル扱いすると、その画像が公開ベンチマークや図表に載ってしまう
    （CLAUDE.md §2.1、ip_api_policy.yaml `allow_publish_ip_images: false`）ので、疑わしきは IP 側に倒す。
    True になるのは、商品マスタが「オリジナル」一択で、かつ IP マスタが IP と言っていない SKU だけ。
    「ライセンス」「その他」、区分なし、区分が複数、両マスタの矛盾はすべて False（＝IP 扱い）。
    """
    names = skus["item_class_names"]
    master_original = names.map(
        lambda v: v is not None and len(v) == 1 and v[0] == ORIGINAL_LABEL
    )
    ip_says_ip = skus["is_ip_draft"].fillna(False).astype(bool)
    return (master_original & ~ip_says_ip).astype(bool)


SKU_COLS = [
    "class_id", "color_id", "n_color_ids", "n_sizes", "release_date", "first_transaction_date",
    "last_sale_date", "has_preorder", "content_id", "n_content_ids", "content_name",
    "creator_name", "design_category_name", "season_name", "item_class_name_1",
    "item_class_name_3", "item_category_name", "price_proper_max", "price_proper_min",
    "units_lifetime", "revenue_lifetime",
]

WEEKLY_COLS = [
    "class_id", "week_index", "units", "units_fullprice", "units_sale", "units_ec",
    "units_store_proper", "units_outlet", "units_preorder", "units_returned", "revenue",
    "n_stores_selling",
]


DATE_COLS = ("release_date", "first_transaction_date", "last_sale_date")


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery クライアントが返す db_dtypes.DateDtype を素の datetime64 に落とす。

    そのまま parquet に書くと読み戻しに db_dtypes の import が必要になり、
    下流スクリプトが BigQuery 依存になってしまうため。
    """
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _observation_end(client: bigquery.Client, orders: str) -> pd.Timestamp:
    """テーブルに入っている最終受注日。右側打ち切りの判定に使う。"""
    sql = f"SELECT MAX(DATE(order_date)) AS d FROM `{orders}`"
    return pd.Timestamp(run_query(client, sql, label="observation_end").loc[0, "d"])


def build_outcomes(weekly: pd.DataFrame, skus: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """§3.4 のアウトカムのうち、分母（配荷店舗数・初回生産数）を必要としないものを作る。

    y_units_per_store_* と y_sellthrough_* は A4 / A5 が解決するまで作れない。
    """
    post = weekly[weekly["week_index"] >= 0]
    out = skus[["class_id"]].copy()

    for w in sorted(windows):
        win = post[post["week_index"] < w]
        agg = win.groupby("class_id").agg(
            units=("units", "sum"),
            units_fullprice=("units_fullprice", "sum"),
            units_ec=("units_ec", "sum"),
            units_store=("units_store_proper", "sum"),
            units_outlet=("units_outlet", "sum"),
            revenue=("revenue", "sum"),
            n_stores_selling=("n_stores_selling", "max"),
            weeks_with_sales=("units", lambda s: int((s > 0).sum())),
        )
        agg.columns = [f"{c}_{w}w" for c in agg.columns]
        out = out.merge(agg, left_on="class_id", right_index=True, how="left")

    # 予約分（week_index < 0）は発売後アウトカムに混ぜず別列にする
    pre_rows = weekly[weekly["week_index"] < 0]
    pre = pre_rows.groupby("class_id")["units"].sum()
    out["units_prerelease"] = out["class_id"].map(pre).fillna(0)
    # 予約の 99.1% は EC なので、EC 系アウトカムには EC 分だけを足す
    pre_ec = pre_rows.groupby("class_id")["units_ec"].sum()
    out["units_ec_prerelease"] = out["class_id"].map(pre_ec).fillna(0)

    count_cols = [c for c in out.columns if c != "class_id"]
    out[count_cols] = out[count_cols].fillna(0)

    for w in sorted(windows):
        u = out[f"units_{w}w"]
        out[f"y_units_{w}w"] = u
        # 予約込みの「立ち上がり需要」。予約は消費者の需要そのものなので、
        # 除くと予約枠を持った SKU（26%）だけ需要が過小に出る。中央値で 27% 分。
        # どちらを主アウトカムにするかは questions.md A26 で確認中。
        out[f"y_units_launch_{w}w"] = u + out["units_prerelease"]
        # 返品超過で負になる僅かな SKU は 0 に潰してから log1p
        out[f"y_log1p_units_{w}w"] = np.log1p(u.clip(lower=0))

        # --- 主アウトカム系列（EC チャネル）---
        # 素の販売点数は配荷幅に支配される（ρ(units_8w, 販売店舗数) = 0.865）が、
        # EC は全国一律露出なので 0.541 まで落ちる。残差は店舗ハローではなく共通原因
        # （店舗配荷前に立つ予約需要のほうが店舗数との ρ が高い: 0.595 > 0.469）。
        # docs/outcome_decision.md / docs/decisions_log.md 2026-09-18
        ec = out[f"units_ec_{w}w"]
        out[f"y_units_ec_{w}w"] = ec
        # 予約込み。予約は発売時 EC 需要の中央値 62%（予約ありの 194 点）を占めるので、
        # 除くと該当 SKU の需要シグナルの過半が落ちる。予約は「画像・商品名・価格だけを見た
        # 購買判断」であり §5.2 の刺激条件と構造が一致するため、ground truth として残す。
        out[f"y_units_ec_launch_{w}w"] = ec + out["units_ec_prerelease"]
        out[f"y_log1p_units_ec_launch_{w}w"] = np.log1p(
            out[f"y_units_ec_launch_{w}w"].clip(lower=0)
        )
        # 正価比率: 分母が 0（= 窓内の純販売が 0）の SKU は NaN のまま残す
        out[f"y_fullprice_share_{w}w"] = (
            out[f"units_fullprice_{w}w"].where(u > 0) / u.where(u > 0)
        )
        out[f"ec_share_{w}w"] = out[f"units_ec_{w}w"].where(u > 0) / u.where(u > 0)

    return out


class SkuRow(BaseModel):
    """skus.parquet の 1 行。CLAUDE.md §6 のスキーマ検証用。"""

    model_config = ConfigDict(extra="ignore")

    class_id: str = Field(min_length=9, max_length=9)
    color_id: str
    n_sizes: int = Field(ge=1)
    release_date: datetime
    last_sale_date: datetime
    has_preorder: bool
    price_proper_max: int | None = Field(default=None, ge=0)  # 受注明細由来。特徴量に使わないこと
    price_list: int | None = Field(default=None, ge=0)
    units_lifetime: int
    observed_weeks: int = Field(ge=0)

    @field_validator("class_id")
    @classmethod
    def _digits(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError(f"class_id が数字のみではありません: {v!r}")
        return v


class WeeklyRow(BaseModel):
    """sales_weekly.parquet の 1 行。"""

    model_config = ConfigDict(extra="ignore")

    class_id: str = Field(min_length=9, max_length=9)
    week_index: int = Field(ge=WEEK_MIN, le=WEEK_MAX)
    units: int
    units_fullprice: int
    units_returned: int = Field(ge=0)
    revenue: int
    n_stores_selling: int = Field(ge=0)


def _validate_rows(df: pd.DataFrame, model: type[BaseModel], name: str) -> None:
    """全行を pydantic モデルに通す。最初の 5 件のエラーをまとめて報告する。"""
    errors: list[str] = []
    for i, rec in enumerate(df.to_dict("records")):
        try:
            model.model_validate(rec)
        except ValidationError as exc:
            errors.append(f"{name}[{i}]: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
            if len(errors) >= 5:
                break
    if errors:
        raise ValueError(f"{name} のスキーマ検証に失敗:\n  " + "\n  ".join(errors))
    logger.info("%s: %s 行の pydantic 検証を通過", name, f"{len(df):,}")


def validate(skus: pd.DataFrame, weekly: pd.DataFrame, outcomes: pd.DataFrame) -> list[str]:
    """CLAUDE.md §6 のスキーマ検証。致命的な不整合は例外、注意点は warning として返す。"""
    warnings: list[str] = []

    _validate_rows(skus, SkuRow, "skus")
    _validate_rows(weekly, WeeklyRow, "sales_weekly")

    if skus["class_id"].duplicated().any():
        raise ValueError("skus の class_id が一意ではありません")
    if weekly.duplicated(["class_id", "week_index"]).any():
        raise ValueError("sales_weekly の (class_id, week_index) が一意ではありません")
    if set(weekly["class_id"]) - set(skus["class_id"]):
        raise ValueError("sales_weekly に skus に無い class_id があります")
    if skus["release_date"].isna().any():
        raise ValueError("release_date が欠損している SKU があります")

    if not (skus["class_id"].str.len() == 9).all():
        warnings.append("class_id の桁数が 9 でない行があります（画像 URL 生成に影響）")
    multi = int((skus["n_color_ids"] > 1).sum())
    if multi:
        warnings.append(f"1 class_id に複数 color_id を持つ SKU が {multi} 件（画像 URL が一意に決まらない）")
    multi_ip = int((skus["n_content_ids"] > 1).sum())
    if multi_ip:
        warnings.append(f"1 class_id に複数 content_id を持つ SKU が {multi_ip} 件")
    neg = int((outcomes["units_8w"] < 0).sum())
    if neg:
        warnings.append(f"8 週純販売数が負（返品超過）の SKU が {neg} 件")
    zero = int((outcomes["units_8w"] <= 0).sum())
    if zero:
        warnings.append(f"8 週純販売数が 0 以下の SKU が {zero} 件（正価比率は NaN）")
    return warnings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    client = get_client(settings)

    orders = require(settings, "bigquery.orders")
    category = require(settings, "scope.primary_category")
    years = require(settings, "scope.release_years")
    windows = require(settings, "outcomes.windows_weeks")
    start, end = f"{min(years)}-01-01", f"{max(years) + 1}-01-01"

    obs_end = _observation_end(client, orders)
    logger.info("観測終端: %s", obs_end.date())

    params = [
        bigquery.ScalarQueryParameter("category", "STRING", category),
        bigquery.ScalarQueryParameter("period_start", "DATE", start),
        bigquery.ScalarQueryParameter("period_end", "DATE", end),
        bigquery.ScalarQueryParameter("week_min", "INT64", WEEK_MIN),
        bigquery.ScalarQueryParameter("week_max", "INT64", WEEK_MAX),
        bigquery.ArrayQueryParameter("exclude_hanbai", "STRING", ["受注生産"]),
        bigquery.ArrayQueryParameter(
            "exclude_production", "STRING", require(settings, "scope.exclude_production_kbn")
        ),
    ]
    panel = run_query(
        client, PANEL_SQL.format(orders=orders), params, label="panel", max_scan_gb=20.0
    )

    panel = _normalize_dates(panel)
    skus = panel[SKU_COLS].drop_duplicates("class_id").reset_index(drop=True)
    weekly = panel[WEEKLY_COLS].reset_index(drop=True)

    # content_id の補完が IP 属性の結合より先に要るので、商品マスタを先に引く
    item_class = run_query(
        client,
        ITEM_CLASS_SQL.format(item_master=require(settings, "bigquery.item_master")),
        label="item_class",
        max_scan_gb=1.0,
    )
    skus = skus.merge(item_class, on="class_id", how="left")
    skus = fill_content_from_item_master(skus)

    ip = run_query(
        client,
        IP_SQL.format(
            ip_master=require(settings, "bigquery.ip_master"),
            ip_tags=require(settings, "bigquery.ip_tags"),
        ),
        label="ip_master",
        max_scan_gb=1.0,
    ).drop_duplicates("content_id")
    # ip_master には content_id が NULL の行がある。pandas の merge は NULL 同士を結合するので、
    # 落とさないと content_id なしの SKU 全件に is_ip_draft=True 等が付く（2026-09-28 に発見）。
    ip = ip[ip["content_id"].notna()]
    skus = skus.merge(ip, on="content_id", how="left")
    skus["item_class_name"] = skus["item_class_names"].map(
        lambda v: v[0] if v is not None and len(v) == 1 else None
    )
    skus["is_original"] = derive_is_original(skus)
    # is_original の後に置く。<series> / <series> は大半がオリジナルなので、名前だけで補うと
    # オリジナルにまで ip_scale が付く（2026-09-29 に 455 点で発生）
    skus = apply_ip_supplement(skus, REPO_ROOT / "config" / "ip_tags_supplement.yaml")
    skus = skus.drop(columns="item_class_names")

    # 右側打ち切り: 発売から windows 週が観測終端を超えていないか
    max_w = max(windows)
    skus["observed_weeks"] = (
        (obs_end - pd.to_datetime(skus["release_date"])).dt.days // 7
    ).astype(int)
    skus["fully_observed"] = skus["observed_weeks"] >= max_w

    outcomes = build_outcomes(weekly, skus, windows)

    warnings = validate(skus, weekly, outcomes)

    outdir = REPO_ROOT / require(settings, "paths.processed")
    outdir.mkdir(parents=True, exist_ok=True)
    skus.to_parquet(outdir / "skus.parquet", index=False)
    weekly.to_parquet(outdir / "sales_weekly.parquet", index=False)
    outcomes.to_parquet(outdir / "outcomes.parquet", index=False)

    logger.info("skus=%s sales_weekly=%s outcomes=%s", len(skus), len(weekly), len(outcomes))
    for w in warnings:
        logger.warning("VALIDATION: %s", w)
    logger.info("出力先: %s", outdir)


if __name__ == "__main__":
    main()
