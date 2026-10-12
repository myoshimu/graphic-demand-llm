#!/usr/bin/env python3
"""ベースライン B0–B4 の学習・評価（CLAUDE.md §5.1 / §5.3）。引数なしで再実行可能。

出力
  - data/processed/baseline_predictions.parquet  テスト期間の予測（ターゲット尺度）
  - reports/phase1_baselines.md                   指標と 95% クラスタ・ブートストラップ区間
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.baselines import models as B  # noqa: E402
from src.eval.metrics import all_metrics, cluster_bootstrap  # noqa: E402
from src.features.clip_embed import embedding_config, embedding_matrix, load_cache  # noqa: E402

log = structlog.get_logger()
PROC = REPO / "data" / "processed"


def load_frame(settings: dict, bcfg: dict, years: list[int] | None = None,
               graphic_ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """母集団の分析用フレーム。years の既定は学習＋テスト期間（確認用の 2026 年は scripts/run_confirm.py で別に読む）。"""
    y_raw = settings["outcomes"]["primary"]
    y_tgt = settings["outcomes"][bcfg["target"]]
    split = settings["evaluation"]["split"]
    years = years or [*split["train"], *split["test"]]
    s = pd.read_parquet(PROC / "skus.parquet")
    s = s[pd.to_datetime(s["release_date"]).dt.year.isin(years)]
    pop = pd.read_parquet(PROC / "population.parquet")
    o = pd.read_parquet(PROC / "outcomes.parquet")[["class_id", y_raw, y_tgt]]
    g = graphic_ids if graphic_ids is not None else pd.read_parquet(PROC / "graphic_ids.parquet")
    g = g[["class_id", "color_id", "graphic_id"]]
    man = pd.read_parquet(REPO / "data" / "cache" / "image_manifest.parquet")[["class_id", "color_id", "image_sha256"]]
    d = (
        s.merge(pop.loc[pop["in_population"].fillna(False).astype(bool), ["class_id", "color_id"]], on=["class_id", "color_id"])
        .merge(o, on="class_id", how="left")
        .merge(g, on=["class_id", "color_id"], how="left")
        .merge(man, on=["class_id", "color_id"], how="left")
    )
    if d[[y_raw, y_tgt, "graphic_id", "image_sha256"]].isna().any().any():
        raise ValueError("母集団にアウトカム／graphic_id／画像の欠損がある")
    rel = pd.to_datetime(d["release_date"])
    d["year"], d["release_month"] = rel.dt.year, rel.dt.month
    f = bcfg["features"]
    d["is_original"] = d["is_original"].astype(bool)
    d["ip_scale"] = np.where(
        d["is_original"], f["ip_scale_original_label"], d["ip_scale"].fillna(f["ip_scale_missing_label"]).astype(str)
    )
    d["split"] = np.select([d["year"].isin(split["train"]), d["year"].isin(split["test"])], ["train", "test"], "none")
    return d.rename(columns={y_raw: "y_raw", y_tgt: "y_tgt"})


def main() -> None:
    settings = yaml.safe_load((REPO / "config" / "settings.yaml").read_text(encoding="utf-8"))
    bcfg = yaml.safe_load((REPO / "config" / "baselines.yaml").read_text(encoding="utf-8"))
    models_cfg = yaml.safe_load((REPO / "config" / "models.yaml").read_text(encoding="utf-8"))
    seed = int(settings["random_seed"])
    ev = settings["evaluation"]

    d = load_frame(settings, bcfg)
    tr, te = d[d["split"] == "train"].reset_index(drop=True), d[d["split"] == "test"].reset_index(drop=True)
    cache = load_cache(REPO / embedding_config(models_cfg, "clip_vit_l14")["output"])
    data = B.Data(
        X_tr=tr, X_te=te,
        E_tr=embedding_matrix(cache, tr["image_sha256"]), E_te=embedding_matrix(cache, te["image_sha256"]),
        y_tr=tr["y_tgt"].to_numpy(), groups_tr=tr["graphic_id"].to_numpy(),
    )
    log.info("baselines", n_train=len(tr), n_test=len(te))

    preds: dict[str, np.ndarray] = {}
    params: dict[str, dict] = {}
    preds["B0_constant"] = B.b0_constant(data)
    for name, (p, hp) in B.b1_meta(data, bcfg, seed).items():
        preds[f"B1_{name}"], params[f"B1_{name}"] = p, hp
    for name, (p, hp) in B.b1_meta(data, bcfg, seed, drop=("ip_scale",)).items():
        preds[f"B1_noipscale_{name}"], params[f"B1_noipscale_{name}"] = p, hp
    for name, (p, hp) in B.b2_image(data, bcfg, seed).items():
        preds[f"B2_{name}"], params[f"B2_{name}"] = p, hp
    preds["B3_ip_scale"] = B.b3_ip_scale(data)
    preds["B4_knn"], params["B4_knn"] = B.b4_knn(data, bcfg)
    log.info("fitted", models=list(preds))

    out = te[["class_id", "color_id", "graphic_id", "is_original", "ip_scale", "y_raw", "y_tgt"]].copy()
    for k, v in preds.items():
        out[f"pred_{k}"] = v
    out.to_parquet(PROC / "baseline_predictions.parquet", index=False)

    fracs = ev["precision_at_k"]
    names = list(preds)

    def evaluate(df: pd.DataFrame) -> dict[str, float]:
        y = df["y_raw"].to_numpy(dtype=float)
        r: dict[str, float] = {}
        for n in names:
            for mk, mv in all_metrics(y, df[f"pred_{n}"].to_numpy(), fracs).items():
                r[f"{n}|{mk}"] = mv
        # B1 に対する上積み（同じリサンプル上の差なので対応あり）
        for n in names:
            r[f"{n}|d_spearman_vs_B1"] = r[f"{n}|spearman"] - r["B1_lgbm|spearman"]
        return r

    bs = ev["bootstrap"]
    rows = []
    # 学習・テストをまたぐグラフィック（リーク候補）を除いた感度分析も並べる
    span = d.groupby("graphic_id")["split"].nunique()
    span_ids = set(span[span > 1].index)
    subsets = {
        "all": out,
        "original": out[out["is_original"]],
        "ip": out[~out["is_original"]],
        "no_span_graphic": out[~out["graphic_id"].isin(span_ids)],
    }
    for sub, df in subsets.items():
        point = pd.Series(evaluate(df))
        ci = cluster_bootstrap(df.reset_index(drop=True), evaluate, bs["cluster_column"], bs["n_resamples"], seed)
        t = pd.DataFrame({"est": point}).join(ci)
        t["subset"], t["n"] = sub, len(df)
        rows.append(t)
    res = pd.concat(rows).rename_axis("key").reset_index()
    res[["model", "metric"]] = res["key"].str.split("|", expand=True)
    res.drop(columns="key").to_parquet(PROC / "baseline_metrics.parquet", index=False)
    write_report(res, params, names, len(tr), len(te), subsets, span_ids, bs)


def write_report(res, params, names, n_tr, n_te, subsets, span_ids, bs) -> None:
    fmt = lambda r: "–" if pd.isna(r.est) else f"{r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}]"  # noqa: E731
    metrics = ["spearman", "precision_at_10pct", "precision_at_20pct", "ndcg_at_10pct", "lift_top10pct"]
    L = [
        "# フェーズ1 ベースライン B0–B4（テスト期間）",
        "",
        "`make baselines`（scripts/run_baselines.py）が自動生成。手で編集しないこと。",
        f"学習 {n_tr} 点（2023–24 発売）→ テスト {n_te} 点（2025 発売）。評価は主アウトカムの生の値。",
        f"[ ] は graphic_id クラスタ・ブートストラップ（{bs['n_resamples']} 回）の 95% 区間。",
        "Precision@k のランダム期待値は k（10% → 0.100）。lift はランダムで 1.0。",
        "",
    ]
    for sub, df in subsets.items():
        t = res[res["subset"] == sub]
        L += [f"## {sub}（n={len(df)}）", "", "| モデル | " + " | ".join(metrics) + " | Δρ vs B1 |", "|---" * (len(metrics) + 2) + "|"]
        for n in names:
            tm = t[t["model"] == n].set_index("metric")
            cells = [fmt(tm.loc[m]) for m in metrics + ["d_spearman_vs_B1"]]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L.append("")
    L += [
        "## 注記",
        "",
        f"- `no_span_graphic`：学習期間にも同じグラフィックがある {len(span_ids)} 個を除いたテスト SKU（リーク感度）。",
        "- `ip_scale` は 2025 年の IP で 124 点欠損（questions.md A32）。`unknown` 水準は学習期間にほぼ無いので、"
        "B1/B3 の IP 部分は A32 決着まで暫定。`B1_noipscale` は ip_scale を落とした版。",
        "- B3 は Google Trends 未取得のため §5.1 の代替どおり ip_scale の学習期間平均。",
        "- ハイパーパラメータ（学習期間内 GroupKFold で選択）：",
    ]
    L += [f"  - {k}: {v}" for k, v in params.items()]
    (REPO / "reports" / "phase1_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/phase1_baselines.md")


if __name__ == "__main__":
    main()
