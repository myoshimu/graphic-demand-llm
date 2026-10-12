"""商品画像の取得（CLAUDE.md §3.1 / §7-4）。

制約（config/settings.yaml の `images` で管理。コードにハードコードしない）:
  - **1 req/sec 以下**。並列度 1。上げないこと
  - リトライ（指数バックオフ）
  - **取得済みは再取得しない**（`skip_if_cached`）

URL は `{class_id}.{color_id}.-_1.jpg`。**class_id は 9 桁ゼロ埋めのまま使うこと。**
先頭ゼロが落ちると CDN は 403 を返す（実測確認済み。2026-09-15 に `000000000` → 200、
`19000749` → 403）。以前 A19 として「画像の 13% が欠測」としていたのは、CSV 側で
先頭ゼロが落ちた class_id から URL を組んでいたことによる取得ミスだった。

出力:
  - 画像本体: `data/cache/images/{class_id}.{color_id}.jpg`
  - マニフェスト: `data/cache/image_manifest.parquet`
    （class_id, color_id, url, status, http_code, image_path, image_sha256, n_bytes,
      width, height, fetched_at, n_attempts）
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

from .bq import REPO_ROOT, load_settings, require

logger = logging.getLogger(__name__)

MANIFEST_NAME = "image_manifest.parquet"


def build_url(template: str, class_id: str, color_id: str) -> str:
    """§3.1 の URL テンプレートを埋める。

    `class_id` / `color_id` は**ゼロ埋めを保ったまま**渡すこと。int を経由させない。
    """
    if not class_id.isdigit() or len(class_id) != 9:
        raise ValueError(f"class_id は 9 桁の数字でなければなりません: {class_id!r}")
    return template.format(product_id=class_id, color_code=color_id)


class RateLimiter:
    """直近の呼び出しから最低 `1/rps` 秒あける。"""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


def fetch_one(
    url: str,
    dest: Path,
    *,
    user_agent: str,
    timeout: float,
    max_retries: int,
    backoff: float,
    limiter: RateLimiter,
) -> dict[str, Any]:
    """1 枚取得する。404/403 はリトライしない（存在しないものは何度叩いても無い）。"""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(1, max_retries + 1):
        limiter.wait()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return {"status": "ok", "http_code": 200, "n_bytes": len(body), "n_attempts": attempt}
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):  # 恒久的な不在。リトライしない
                return {"status": "missing", "http_code": exc.code, "n_bytes": 0,
                        "n_attempts": attempt}
            logger.warning("HTTP %s (%s/%s) %s", exc.code, attempt, max_retries, url)
            last_code = exc.code
        except Exception as exc:  # タイムアウト・接続断など
            logger.warning("%s (%s/%s) %s", type(exc).__name__, attempt, max_retries, url)
            last_code = -1
        if attempt < max_retries:
            time.sleep(backoff * attempt)
    return {"status": "error", "http_code": last_code, "n_bytes": 0, "n_attempts": max_retries}


def _image_dims(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None, None


def fetch_all(pairs: Iterable[tuple[str, str]], settings: dict[str, Any]) -> pd.DataFrame:
    cfg = settings["images"]
    imgdir = REPO_ROOT / require(settings, "paths.images")
    cachedir = REPO_ROOT / require(settings, "paths.cache")
    manifest_path = cachedir / MANIFEST_NAME

    done: set[tuple[str, str]] = set()
    prior = pd.DataFrame()
    if cfg.get("skip_if_cached", True) and manifest_path.exists():
        prior = pd.read_parquet(manifest_path)
        ok = prior[prior["status"] == "ok"]
        done = {
            (r.class_id, r.color_id)
            for r in ok.itertuples()
            if (imgdir / f"{r.class_id}.{r.color_id}.jpg").exists()
        }
        logger.info("キャッシュ済み %s 件をスキップ", f"{len(done):,}")

    limiter = RateLimiter(float(cfg["rate_limit_rps"]))
    rows: list[dict[str, Any]] = []
    todo = [p for p in pairs if p not in done]
    logger.info("取得対象 %s 件（1 req/sec なので約 %.0f 分）", f"{len(todo):,}", len(todo) / 60)

    for i, (class_id, color_id) in enumerate(todo, 1):
        url = build_url(cfg["url_template"], class_id, color_id)
        dest = imgdir / f"{class_id}.{color_id}.jpg"
        res = fetch_one(
            url, dest,
            user_agent=cfg["user_agent"],
            timeout=float(cfg["timeout_seconds"]),
            max_retries=int(cfg["max_retries"]),
            backoff=float(cfg["backoff_seconds"]),
            limiter=limiter,
        )
        sha, width, height = None, None, None
        if res["status"] == "ok":
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            width, height = _image_dims(dest)
        rows.append({
            "class_id": class_id, "color_id": color_id, "url": url,
            "image_path": str(dest.relative_to(REPO_ROOT)) if res["status"] == "ok" else None,
            "image_sha256": sha, "width": width, "height": height,
            "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None), **res,
        })
        if i % 100 == 0:
            logger.info("%s/%s", i, len(todo))

    new = pd.DataFrame(rows)
    manifest = (
        pd.concat([prior, new], ignore_index=True)
        .drop_duplicates(["class_id", "color_id"], keep="last")
        .reset_index(drop=True)
        if not prior.empty else new
    )
    cachedir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    return manifest


def main(limit: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    skus = pd.read_parquet(REPO_ROOT / require(settings, "paths.processed") / "skus.parquet")
    pairs = list(zip(skus["class_id"], skus["color_id"]))
    if limit:
        pairs = pairs[:limit]

    manifest = fetch_all(pairs, settings)
    counts = manifest["status"].value_counts().to_dict()
    logger.info("マニフェスト %s 件: %s", f"{len(manifest):,}", counts)
    ok = manifest[manifest["status"] == "ok"]
    if len(ok):
        logger.info(
            "成功率 %.1f%% / 平均 %.0f KB / 解像度 %s",
            100 * len(ok) / len(manifest),
            ok["n_bytes"].mean() / 1024,
            ok.groupby(["width", "height"]).size().idxmax(),
        )
    failed = manifest[manifest["status"] != "ok"]
    if len(failed):
        logger.warning("失敗 %s 件（先頭 10 件）:\n%s", len(failed),
                       failed[["class_id", "color_id", "status", "http_code"]].head(10).to_string())


if __name__ == "__main__":
    main()
