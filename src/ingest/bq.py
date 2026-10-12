"""BigQuery アクセスの共通ヘルパ。

CLAUDE.md §6 に従い、接続先・プロジェクトは config/settings.yaml から読む。
`wsdm.orders` はパーティション・クラスタリングとも無い 14.5 GB のテーブルなので、
(1) SELECT * を禁止し、(2) 実行前に必ず dry-run でスキャン量を出す。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from google.cloud import bigquery

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

_SELECT_STAR = re.compile(r"select\s+\*", re.IGNORECASE)


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """config/settings.yaml を読む。"""
    with open(path or SETTINGS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def require(settings: dict[str, Any], dotted_key: str) -> Any:
    """未確定（null）の設定値を読んだら例外を投げる。

    CLAUDE.md §2.5「推測でスキーマや閾値を決めない」に対応。既定値で黙って走らせない。
    """
    node: Any = settings
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config/settings.yaml に {dotted_key} がありません")
        node = node[part]
    if node is None:
        raise ValueError(
            f"config/settings.yaml の {dotted_key} が未確定 (null) です。"
            " docs/questions.md を確認し、確定してから実行してください。"
        )
    return node


def get_client(settings: dict[str, Any]) -> bigquery.Client:
    return bigquery.Client(project=require(settings, "bigquery.billing_project"))


def run_query(
    client: bigquery.Client,
    sql: str,
    params: list[bigquery.ScalarQueryParameter] | None = None,
    *,
    label: str = "query",
    max_scan_gb: float = 20.0,
) -> pd.DataFrame:
    """dry-run でスキャン量を確認してから本実行する。

    max_scan_gb を超える場合は実行せず例外にする（列の絞り忘れの検出用）。
    """
    if _SELECT_STAR.search(sql):
        raise ValueError("SELECT * は禁止です（config: bigquery.forbid_select_star）。列を明示してください。")

    job_config = bigquery.QueryJobConfig(query_parameters=params or [], dry_run=True, use_query_cache=False)
    dry = client.query(sql, job_config=job_config)
    scan_gb = dry.total_bytes_processed / 1e9
    logger.info("[%s] dry-run: %.2f GB (約 $%.3f)", label, scan_gb, scan_gb / 1000 * 6.25)
    if scan_gb > max_scan_gb:
        raise RuntimeError(f"[{label}] スキャン量 {scan_gb:.1f} GB が上限 {max_scan_gb} GB を超えています")

    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    df = client.query(sql, job_config=job_config).to_dataframe()
    logger.info("[%s] %s 行を取得", label, f"{len(df):,}")
    return df
