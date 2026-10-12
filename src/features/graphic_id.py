"""`graphic_id` の導出（CLAUDE.md §3.3 / §7-6）。

§3.3 の定義：「同一イラストの横展開を束ねる ID」。マスタに相当する列は無い
（`content_id` はコレクション／IP 単位で粗すぎる。T シャツ 836 点が 64 個の
`content_id` にしか分かれず、Dragon Quest だけで 105 点ある）。よって画像から推定する。

---
## 測って分かったこと（詳細は reports/phase0_graphic_id.md）

**単一の知覚ハッシュ閾値では原理的に解けない。** このコホートには次の 2 つが同居する。

  (a) **同一イラスト × ボディ色違い**（束ねたい）→ 画素上は**遠い**。
      実測で pHash 距離 38〜90（例：Suica ペンギン 白 vs 青 = 38、
      BACK TO BASIC 黒 vs 緑 = 60、<series> 生成り vs カーキ = 68）
  (b) **文字だけ違う別デザイン × 同一ボディ**（分けたい）→ 画素上は**近い**。
      `Dragon Quest DQg Jumon100 Tshirt`（105 点、呪文名だけが違う黒 T）が該当し、
      「メラ」vs「イオ」= 5、「リベホイミ」vs「リベホイム」= 38

(a) の最小値と (b) の最大値が重なるので、閾値をどこに置いても片方が壊れる。
前処理（autocontrast / equalize / FIND_EDGES × phash / dhash / ahash の 15 通り）を
総当たりしても正のマージンは得られなかった（最良で margin = 0）。

## そこで採る方式：**ブロッキング ＋ 人手確定**

1. **クロップ**：胸元プリント領域 `CHEST_CROP` に寄せてからハッシュする。
   全身だと生地の面積が支配的になり、別デザインの黒 T どうしが距離 0 になる
   （実測：全身 pHash で「ルカナン」vs「メダパニ」= 0 → 胸元クロップで 90）。
2. **ブロッキング**：辺を張る候補を
   **「同一 `content_id` かつ `color_id` が異なる」ペアに限定**する。
   横展開は同じコレクション内で色を変えて出るという商習慣に沿う制約であり、
   同時に (b) の Dragon Quest 系（ほぼ全点が `color_id='100'`）を機械的に排除できる。
3. **距離で候補を絞る**（`threshold`）。ここまでで 666 点 → 候補クラスタ 29 個まで落ちる。
4. **人手で確定**する。候補クラスタは数十個しかないので目視で足りる。
   却下したペアは `config/graphic_id_review.yaml` の `reject` に書く。
   ここを通さない限り `assign()` は暫定フラグ付きの結果しか返さない。

閾値だけで自動確定させない理由は、実測した精度がそれを許さないから。
T=96 の 29 クラスタを目視した限り、明らかな誤結合が 1/4 ほど混じる。
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from ..ingest.bq import REPO_ROOT, load_settings, require

logger = logging.getLogger(__name__)

HASH_SIZE = 16  # 256 bit。既定の 8（64 bit）だと 666 点でも偶然衝突が出る

# 商品画像は 1125x1575 の吊るし写真。胸元プリントはこの範囲にほぼ収まる。
# ここを外して全身を見ると、生地の面積が支配的になって別デザインが衝突する。
CHEST_CROP = (0.33, 0.27, 0.67, 0.50)  # (left, top, right, bottom) の相対座標
# 比較用に残している広めのクロップ。分離は CHEST より明確に悪い
WIDE_CROP = (0.22, 0.20, 0.78, 0.62)

REVIEW_PATH = REPO_ROOT / "config" / "graphic_id_review.yaml"


def _load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _crop(im: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = im.size
    l, t, r, b = box
    return im.crop((int(w * l), int(h * t), int(w * r), int(h * b)))


HASHERS: dict[str, Callable[[Image.Image], imagehash.ImageHash]] = {
    # 本命。胸元クロップ × pHash
    "phash_chest": lambda im: imagehash.phash(_crop(im, CHEST_CROP), hash_size=HASH_SIZE),
    # 比較用（reports/phase0_graphic_id.md に分離性能を載せるため）
    "phash_wide": lambda im: imagehash.phash(_crop(im, WIDE_CROP), hash_size=HASH_SIZE),
    "phash_full": lambda im: imagehash.phash(im, hash_size=HASH_SIZE),
    "dhash_chest": lambda im: imagehash.dhash(_crop(im, CHEST_CROP), hash_size=HASH_SIZE),
}

PRIMARY_HASH = "phash_chest"


def compute_hashes(manifest: pd.DataFrame) -> pd.DataFrame:
    """取得済み画像すべてについて `HASHERS` を計算する。"""
    ok = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ok.itertuples(), 1):
        path = REPO_ROOT / r.image_path
        try:
            im = _load(path)
        except Exception as exc:
            logger.warning("読み込み失敗 %s: %s", path, exc)
            continue
        rec: dict[str, Any] = {"class_id": r.class_id, "color_id": r.color_id}
        for name, fn in HASHERS.items():
            rec[name] = str(fn(im))
        rows.append(rec)
        if i % 200 == 0:
            logger.info("%s/%s", i, len(ok))
    return pd.DataFrame(rows)


def _bits(hexstr: str) -> np.ndarray:
    n = len(hexstr) * 4
    v = int(hexstr, 16)
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def distance_matrix(hashes: pd.DataFrame, column: str) -> np.ndarray:
    """ハミング距離行列（ビット数）。666 点なら一瞬で済む。"""
    a = np.stack([_bits(h) for h in hashes[column]]).astype(np.int16)
    n_bits = a.shape[1]
    same = a @ a.T + (1 - a) @ (1 - a).T
    return (n_bits - same).astype(np.int16)


# ---------------------------------------------------------------- 測定
def separation(hashes: pd.DataFrame, pos: list, neg: list) -> pd.DataFrame:
    """既知の正例（同一イラスト）／負例（別デザイン）でハッシュ種別を比較する。

    `pos` / `neg` は `(class_id_a, class_id_b)` のリスト。
    **margin = min(neg) - max(pos) が正でないと閾値による自動判定は成立しない。**
    """
    idx = {c: i for i, c in enumerate(hashes["class_id"])}
    out = []
    for column in HASHERS:
        d = distance_matrix(hashes, column)
        p = [int(d[idx[a], idx[b]]) for a, b in pos if a in idx and b in idx]
        n = [int(d[idx[a], idx[b]]) for a, b in neg if a in idx and b in idx]
        out.append({
            "hash": column,
            "pos_max": max(p) if p else None, "pos": str(p),
            "neg_min": min(n) if n else None, "neg": str(n),
            "margin": (min(n) - max(p)) if p and n else None,
        })
    return pd.DataFrame(out)


def candidate_clusters(
    hashes: pd.DataFrame, blocks: pd.Series, threshold: int, column: str = PRIMARY_HASH
) -> pd.DataFrame:
    """ブロッキング付き連結成分。

    辺を張る条件：`blocks` が一致し、`color_id` が異なり、距離が `threshold` 以下。
    `blocks` は `hashes` と同じ長さの Series（通常 `content_id`）。欠損は辺を張らない。
    """
    d = distance_matrix(hashes, column)
    n = len(hashes)
    blk = blocks.to_numpy()
    col = hashes["color_id"].to_numpy()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.where(np.triu(d <= threshold, k=1))):
        i, j = int(i), int(j)
        if pd.isna(blk[i]) or blk[i] != blk[j] or col[i] == col[j]:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    out = hashes[["class_id", "color_id"]].copy()
    out["_root"] = [find(i) for i in range(n)]
    return out


def _load_review() -> dict[str, list]:
    """人手確定の結果（分割・却下ペア・強制結合ペア・要確認）を読む。"""
    keys = ("groups", "reject", "force", "uncertain")
    if not REVIEW_PATH.exists():
        return {k: [] for k in keys}
    data = yaml.safe_load(REVIEW_PATH.read_text()) or {}
    return {k: data.get(k) or [] for k in keys}


def assign(
    hashes: pd.DataFrame, blocks: pd.Series, threshold: int, column: str = PRIMARY_HASH
) -> pd.DataFrame:
    """`config/graphic_id_review.yaml` の判定を反映して `graphic_id` を確定する。

    `groups` に載った class_id はハッシュの辺を使わず、書かれた分割どおりに束ねる
    （ペア単位の reject だと、別メンバー経由の連鎖で再結合してしまうため）。
    `reject` のペアは辺を張らず、`force` のペアは距離に関わらず結合する。
    `reviewed` は候補クラスタ（2 点以上）の全メンバーが `groups` で判定済みのとき True。
    `uncertain` に載った class_id は `review_uncertain=True` にする（感度分析用）。
    """
    review = _load_review()
    reject = {tuple(sorted(p)) for p in review["reject"]}
    force = [tuple(p) for p in review["force"]]
    groups = [list(map(str, g)) for g in review["groups"]]
    decided = {c for g in groups for c in g}
    uncertain = {str(c) for c in review["uncertain"]}

    d = distance_matrix(hashes, column)
    n = len(hashes)
    blk, col = blocks.to_numpy(), hashes["color_id"].to_numpy()
    cids = hashes["class_id"].to_numpy()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in zip(*np.where(np.triu(d <= threshold, k=1))):
        i, j = int(i), int(j)
        if pd.isna(blk[i]) or blk[i] != blk[j] or col[i] == col[j]:
            continue
        if cids[i] in decided or cids[j] in decided:
            continue
        if tuple(sorted((cids[i], cids[j]))) in reject:
            continue
        union(i, j)

    idx = {c: i for i, c in enumerate(cids)}
    for a, b in force:
        if a in idx and b in idx:
            union(idx[a], idx[b])
    for g in groups:
        members = [idx[c] for c in g if c in idx]
        for k in members[1:]:
            union(members[0], k)

    cand = candidate_clusters(hashes, blocks, threshold, column)
    csize = cand.groupby("_root")["class_id"].transform("size")
    undecided = set(cand.loc[csize > 1, "class_id"]) - decided

    roots = [find(i) for i in range(n)]
    rep = pd.Series(cids).groupby(roots).transform("min")
    out = hashes[["class_id", "color_id"]].copy()
    out["graphic_id"] = "g" + rep.values
    out["reviewed"] = not undecided
    out["review_uncertain"] = out["class_id"].isin(uncertain)
    sizes = out.groupby("graphic_id").size()
    logger.info(
        "graphic_id %s 個 / class_id %s 点。2 点以上のクラスタ %s 個、最大 %s 点（reviewed=%s）",
        sizes.size, n, int((sizes > 1).sum()), sizes.max(), out["reviewed"].iloc[0],
    )
    if undecided:
        logger.warning(
            "候補クラスタのうち %s 点が config/graphic_id_review.yaml の groups で未判定です。"
            "この graphic_id は**暫定**です", len(undecided),
        )
    return out


# ---------------------------------------------------------------- 人手確認用の出力
def cluster_sheets(
    clusters: pd.DataFrame, labels: pd.Series, outdir: Path, thumb: int = 300
) -> None:
    """候補クラスタを 1 枚ずつ横並び画像にする。ファイル名に点数とコレクション名を入れる。"""
    outdir.mkdir(parents=True, exist_ok=True)
    imgdir = REPO_ROOT / "data" / "cache" / "images"
    df = clusters.assign(_label=labels.to_numpy())
    for root, g in df.groupby("_root"):
        if len(g) < 2:
            continue
        ims = []
        for r in g.itertuples():
            p = imgdir / f"{r.class_id}.{r.color_id}.jpg"
            if not p.exists():
                continue
            im = _load(p)
            im.thumbnail((thumb, thumb))
            ims.append(im)
        if len(ims) < 2:
            continue
        sheet = Image.new("RGB", (sum(i.width for i in ims), max(i.height for i in ims)), "white")
        x = 0
        for im in ims:
            sheet.paste(im, (x, 0))
            x += im.width
        label = str(g["_label"].iloc[0]).replace("/", "_")[:24]
        ids = "__".join(g["class_id"].tolist())[:120]
        sheet.save(outdir / f"n{len(g)}_{label}_{ids}.jpg", quality=85)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    cache = REPO_ROOT / require(settings, "paths.cache")
    reports = REPO_ROOT / require(settings, "paths.reports")
    processed = REPO_ROOT / require(settings, "paths.processed")

    manifest = pd.read_parquet(cache / "image_manifest.parquet")
    hash_path = cache / "image_hashes.parquet"
    need = set(manifest[manifest.status == "ok"].class_id)
    if hash_path.exists():
        hashes = pd.read_parquet(hash_path)
        missing = need - set(hashes.class_id)
        if missing:
            logger.info("未計算 %s 点を追加計算します", len(missing))
            hashes = pd.concat(
                [hashes, compute_hashes(manifest[manifest.class_id.isin(missing)])],
                ignore_index=True)
            hashes.to_parquet(hash_path, index=False)
        if set(HASHERS) - set(hashes.columns):
            logger.info("ハッシュ種別が変わったので再計算します")
            hashes = compute_hashes(manifest)
            hashes.to_parquet(hash_path, index=False)
    else:
        hashes = compute_hashes(manifest)
        hashes.to_parquet(hash_path, index=False)
    hashes = hashes[hashes.class_id.isin(need)].reset_index(drop=True)
    logger.info("ハッシュ %s 点", len(hashes))

    skus = pd.read_parquet(processed / "skus.parquet")[
        ["class_id", "content_id", "content_name"]]
    hashes = hashes.merge(skus, on="class_id", how="left")
    # content_id が無い新規コレクション（2025 年に多い）は名前でブロッキングする
    blocks = hashes["content_id"].fillna("name:" + hashes["content_name"])

    cfg = settings.get("graphic_id") or {}
    threshold = cfg.get("threshold")
    if threshold is None:
        logger.warning("graphic_id.threshold が未設定のため候補生成のみ行います")
        threshold = 96

    clusters = candidate_clusters(hashes, blocks, int(threshold))
    sizes = clusters.groupby("_root").size()
    logger.info("候補クラスタ（2 点以上）%s 個 / 最大 %s 点", int((sizes > 1).sum()), sizes.max())
    cluster_sheets(clusters, hashes["content_name"], reports / "graphic_id_candidates")

    out = assign(hashes, blocks, int(threshold))
    out.to_parquet(processed / "graphic_ids.parquet", index=False)
    logger.info("→ %s", processed / "graphic_ids.parquet")


if __name__ == "__main__":
    main()
