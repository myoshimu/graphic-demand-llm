"""EC 掲載フラグの導出（`docs/questions.md` A19）。

## なぜ必要か

商品画像 `images.example.com/images/item/product_image/{class_id}.{color_id}.-_1.jpg` は
**EC に掲載された商品にしか存在しない**（`reports/phase0_images.md` §2）。
店舗専売品は EC 商品ページが作られないので、画像もアップされない。
2024 年コホートでは 836 点中 170 点（20.3%）がこれに該当し、
その `units_8w` 中央値は 82.5（取得できた側は 524）。**売れない側が体系的に欠ける。**

この欠測を補正するには「EC に掲載されたか」を**画像の有無とは独立に**判定する必要がある。
retailer の商品マスタに掲載フラグ列は無い（2026-09-16 回答）。そこで**閲覧ログ**を使う。

## 判定方法

商品詳細ページの URL は `https://www.retailer.com/item-detail/{class_id}{color_id}`
（12 桁 ＝ 9 桁 ＋ 3 桁）。**このページの初回閲覧が「発売日 + 30 日」以内なら掲載されていた**
とみなす。「1 回でも閲覧された」では判定しない理由は `attach()` の docstring を参照
（実体の無い URL への 404 アクセスを拾ってしまう）。

出典が 2 つあり、期間で繋ぐ：

| 期間 | ソース | 備考 |
|---|---|---|
| 2022-02-01 〜 2025-01-31 | `wsdm.pv_base` | 先行研究で作成済み。DAY パーティション、`page_location` 列を直接持つので安い |
| 2025-02-01 〜 | `asia.events_*`（GA4 エクスポート） | `event_params` から `page_location` を取り出す。スキャン量が大きい |

## 検証

2024 年コホート 836 点で、掲載判定と画像の有無は **836/836 完全一致**した
（掲載 666 ＝ 画像あり 666、未掲載 170 ＝ 画像なし 170、不一致 0）。
CDN と GA という独立な 2 系統が完全に一致するので、
「画像が無い ＝ EC 未掲載」は相関ではなく機構と見てよい。

発売年別の掲載率は 2022: 92.4% / 2023: 87.6% / 2024: 84.0% / 2025: 75.1% / 2026: 66.8% で、
実クロールの取得率（各年 50 点無作為で 96 / 84 / 82 / 74 / 72%）とよく一致する。

## 使い道

1. **クロール前に取得可否が分かる。** 2023–2026 の T シャツ 3,607 点のうち
   画像が取れるのは 2,819 点、と事前に見積もれる。無駄なリクエストを投げない
2. **選択モデルの被説明変数になる。** `ec_listed` を発売前メタデータ（価格・IP・
   カテゴリ・季節）に回帰すれば、欠測を Heckman 型で補正した推定が書ける
3. **交絡として報告できる。** 「画像がある SKU」＝「EC に載せると判断された SKU」。
   その判断自体が社内の需要予想を含むので、精度を報告するときに必ず併記する

**`pv_total` を予測の特徴量に使ってはいけない。** 発売後の閲覧数はアウトカムと同時決定で、
合成パネルが予測すべきものそのもの。ここで使うのは**ゼロか否かの二値だけ**である。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from src.ingest.bq import get_client, load_settings, require, run_query

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "cache" / "ec_listed.parquet"

# item-detail の URL は 12 桁（class_id 9 桁 ＋ color_id 3 桁）
_CODE_RE = r"/item-detail/([0-9]{12})"

# pv_base の被覆期間。ここを外れる分は GA4 から取る（両ソースの境界）
PV_BASE_START = "2022-02-01"
PV_BASE_END = "2025-01-31"

# 掲載判定の窓。初回閲覧が「発売日 + この日数」以内なら掲載されていたとみなす。
# 2024 年コホートの実測：掲載側の初回閲覧は −70〜+2 日、誤検知側は最短 +55 日。
LAUNCH_WINDOW_DAYS = 30


def fetch_pv_base(client: bigquery.Client, start: str, end: str) -> pd.DataFrame:
    """`wsdm.pv_base` から商品詳細ページのビューを集計する。

    パーティション列 `event_dt` で絞るので、1 日あたり約 0.02 GB と安い。
    """
    sql = f"""
    WITH pv AS (
      SELECT REGEXP_EXTRACT(page_location, r'{_CODE_RE}') AS code, event_dt
      FROM `your-gcp-project.wsdm.pv_base`
      WHERE event_dt BETWEEN DATE '{start}' AND DATE '{end}'
        AND page_location LIKE '%/item-detail/%'
    )
    SELECT SUBSTR(code, 1, 9) AS class_id, SUBSTR(code, 10, 3) AS color_id,
           COUNT(*) AS pv_total, MIN(event_dt) AS first_view, MAX(event_dt) AS last_view
    FROM pv WHERE code IS NOT NULL GROUP BY 1, 2
    """
    return run_query(client, sql, label="ec_listed:pv_base", max_scan_gb=60.0)


def fetch_ga4(client: bigquery.Client, start: str, end: str) -> pd.DataFrame:
    """GA4 エクスポート `asia.events_*` から同じものを集計する。

    `event_params` の展開が必要なぶん重い（1 日あたり約 0.8 GB）。
    `_TABLE_SUFFIX` で日付を絞ってもワイルドカード分は全列走査になるので、
    期間を広げる前に必ず dry-run のスキャン量を確認すること。
    """
    sql = f"""
    WITH pv AS (
      SELECT PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) AS event_dt,
             REGEXP_EXTRACT(
               (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location'),
               r'{_CODE_RE}') AS code
      FROM `your-gcp-project.asia.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start.replace('-', '')}' AND '{end.replace('-', '')}'
        AND event_name = 'page_view'
    )
    SELECT SUBSTR(code, 1, 9) AS class_id, SUBSTR(code, 10, 3) AS color_id,
           COUNT(*) AS pv_total, MIN(event_dt) AS first_view, MAX(event_dt) AS last_view
    FROM pv WHERE code IS NOT NULL GROUP BY 1, 2
    """
    return run_query(client, sql, label="ec_listed:ga4", max_scan_gb=700.0)


def merge_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """期間の異なるソースを SKU 単位で足し合わせる。"""
    out = (
        pd.concat(frames)
        .groupby(["class_id", "color_id"], as_index=False)
        .agg(
            pv_total=("pv_total", "sum"),
            first_view=("first_view", "min"),
            last_view=("last_view", "max"),
        )
    )
    for col in ("first_view", "last_view"):
        out[col] = out[col].astype(str)
    out["ec_listed"] = True  # 行が在ること自体がフラグ。join 後の欠損を False とする
    return out


def attach(
    skus: pd.DataFrame,
    listed: pd.DataFrame | None = None,
    *,
    window_days: int = LAUNCH_WINDOW_DAYS,
) -> pd.DataFrame:
    """`skus` に `ec_listed` を付ける。

    **「1 度でも閲覧された」では判定しない。** 発売から半年〜1 年半も経ってから
    PV が 1〜8 だけ付く SKU が 2024 年コホートに 18 点あり、これは実体の無い URL への
    アクセス（古いリンク・クローラ）が 404 ページのビューとして記録されたものと見られる。
    素朴な「PV > 0」だと一致率が 836/836 → 818/836 に落ちる。

    実際に掲載された商品は、**発売の 70 日前〜2 日後**に必ず初回閲覧が付く
    （予約ページが先に公開されるため前倒しになる）。誤検知側は最短でも +55 日なので、
    `release_date + 30 日`で切ると 2024 年コホートでは完全に分離できた。

    `release_date` 列が無い場合は窓を適用できないので PV の有無だけで判定し、
    警告を出す（下流で気づけるように）。
    """
    if listed is None:
        listed = pd.read_parquet(OUT_PATH)
    out = skus.merge(
        listed[["class_id", "color_id", "pv_total", "first_view"]],
        on=["class_id", "color_id"],
        how="left",
    )
    viewed = out["pv_total"].notna()
    if "release_date" not in out.columns:
        logger.warning("release_date が無いので発売窓を適用できない。PV の有無だけで判定する")
        out["ec_listed"] = viewed
        return out

    lag = (pd.to_datetime(out["first_view"]) - pd.to_datetime(out["release_date"])).dt.days
    out["ec_listed"] = (viewed & (lag <= window_days)).fillna(False).astype(bool)
    out["first_view_lag_days"] = lag
    return out


POPULATION_PATH = REPO_ROOT / "data" / "processed" / "population.parquet"


def define_population(skus: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """分析母集団 ＝ **画像あり かつ 発売時に EC 掲載**（questions.md A31、2026-09-28）。

    2024 年は「掲載 ⇔ 画像あり」が 836/836 一致したが、2023・2025 年で 2 種類崩れた。
    どちらも発売 8 週の EC 販売は全件ゼロで、発売時に EC に載っていなかった点は共通：

    - `listed_late`：画像はあるが初回閲覧が発売の 216〜620 日後（2023 年 60 点）。
      閲覧ログが 1 件も無い画像あり SKU 1 点（000000000）もここに入る。
      発売時は EC に無く、後から載ったもの（画像はその時点で撮られたと考えられる）
    - `listed_no_image`：初回閲覧は窓内だが画像なし（2025 年 27 点）。閲覧数は
      最大 8 で、実体の無い URL へのアクセスを拾った誤検知

    判定に売上は使わない（母集団をアウトカムで選ばない）。
    """
    out = attach(skus[["class_id", "color_id", "release_date"]])
    has_image = out["class_id"].isin(manifest.loc[manifest["status"] == "ok", "class_id"])
    out["has_image"] = has_image
    out["in_population"] = out["ec_listed"] & has_image
    reason = pd.Series(pd.NA, index=out.index, dtype="string")
    reason[~out["ec_listed"] & ~has_image] = "not_listed"
    reason[~out["ec_listed"] & has_image] = "listed_late"
    reason[out["ec_listed"] & ~has_image] = "listed_no_image"
    out["exclusion_reason"] = reason
    return out[
        ["class_id", "color_id", "ec_listed", "has_image", "first_view_lag_days",
         "in_population", "exclusion_reason"]
    ]


def build_population() -> None:
    """`make population`：skus と画像マニフェストから population.parquet を作る。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    skus = pd.read_parquet(REPO_ROOT / "data" / "processed" / "skus.parquet")
    manifest = pd.read_parquet(REPO_ROOT / "data" / "cache" / "image_manifest.parquet")
    pop = define_population(skus, manifest)
    pop.to_parquet(POPULATION_PATH, index=False)
    logger.info(
        "母集団 %s / %s 点。除外内訳: %s",
        f"{int(pop['in_population'].sum()):,}",
        f"{len(pop):,}",
        pop["exclusion_reason"].value_counts().to_dict(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    client = get_client(settings)

    ga_start = (pd.Timestamp(PV_BASE_END) + pd.Timedelta(days=1)).date().isoformat()
    ga_end = require(settings, "ec_listed.ga4_end")

    frames = [
        fetch_pv_base(client, PV_BASE_START, PV_BASE_END),
        fetch_ga4(client, ga_start, ga_end),
    ]
    out = merge_sources(frames)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info(
        "EC 掲載 %s SKU / class_id %s 件 を %s に書き出した（%s〜%s）",
        f"{len(out):,}",
        f"{out['class_id'].nunique():,}",
        OUT_PATH,
        out["first_view"].min(),
        out["last_view"].max(),
    )


if __name__ == "__main__":
    main()
