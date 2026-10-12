"""段階 1 の評価：パネルはベースラインに上積みするか → reports/panel_stage1.md。

判定基準（結果を見る前に決めた。decisions_log 2026-10-03（6））：
  主指標は Pro・full のパネルスコア（purchase_intent の平均）を B2_lgbm に足したときの Δρ。
  ベースラインの予測とパネルのスコアを、テスト期間内で graphic_id ごとの 5 分割交差検証による線形回帰で組み合わせる。
  比較対象は、ベースラインだけを同じ手順で回帰し直したもの。
  95% 区間（graphic_id クラスタ・ブートストラップ）の下限が 0 より上なら Go（段階 2 へ）、
  点推定が正で区間が 0 を含むなら「弱い」、点推定が 0 以下なら No-Go（案 3 を主軸に）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import cluster_bootstrap, precision_at_k, spearman  # noqa: E402
from src.ingest.bq import REPO_ROOT  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
BASELINES = ["B1_lgbm", "B2_lgbm"]
PRIMARY = ("B2_lgbm", "full")
FOLDS = 5


def crossfit(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """テスト期間内で graphic_id ごとの GroupKFold による線形の組み合わせ（OOF 予測）。"""
    X, y = df[cols].to_numpy(dtype=float), df["y_tgt"].to_numpy(dtype=float)
    oof = np.empty(len(df))
    for tr, va in GroupKFold(n_splits=FOLDS).split(X, groups=df["graphic_id"]):
        oof[va] = LinearRegression().fit(X[tr], y[tr]).predict(X[va])
    return oof


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    n_bs, seed = int(settings["evaluation"]["bootstrap"]["n_resamples"]), int(settings["random_seed"])
    base = pd.read_parquet(PROC / "baseline_predictions.parquet")
    sc = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet")
    piv = sc.pivot_table(index=["class_id", "color_id"], columns="condition", values="intent_mean").reset_index()
    conds = [c for c in ["full", "image_masked_text"] if c in piv.columns]
    df = base.merge(piv, on=["class_id", "color_id"], how="inner")
    if len(df) != len(base):
        raise ValueError(f"パネルのスコアが揃っていない SKU がある（{len(df)}/{len(base)}）")

    for b in BASELINES:
        df[f"oof_{b}"] = crossfit(df, [f"pred_{b}"])
        for c in conds:
            df[f"oof_{b}+{c}"] = crossfit(df, [f"pred_{b}", c])
        df[f"oof_{b}+both"] = crossfit(df, [f"pred_{b}", *conds])

    def stats(d: pd.DataFrame) -> dict[str, float]:
        y, yt = d["y_raw"].to_numpy(float), d["y_tgt"].to_numpy(float)
        out: dict[str, float] = {}
        for c in conds:
            out[f"rho|panel_{c}"] = spearman(y, d[c].to_numpy())
        for b in BASELINES:
            rb, r2b = spearman(y, d[f"oof_{b}"].to_numpy()), r2(yt, d[f"oof_{b}"].to_numpy())
            out[f"rho|{b}"], out[f"r2|{b}"] = rb, r2b
            out[f"p20|{b}"] = precision_at_k(y, d[f"oof_{b}"].to_numpy(), 0.2)
            for c in [*conds, "both"]:
                p = d[f"oof_{b}+{c}"].to_numpy()
                out[f"drho|{b}+{c}"] = spearman(y, p) - rb
                out[f"dr2|{b}+{c}"] = r2(yt, p) - r2b
                out[f"dp20|{b}+{c}"] = precision_at_k(y, p, 0.2) - out[f"p20|{b}"]
        return out

    lines = ["# 段階 1：パネル（Pro）はベースラインに上積みするか", "",
             "`scripts/eval_panel_stage1.py` が自動生成。テスト期間（2025 年発売）の母集団。",
             "上積みは、テスト期間内の graphic_id 5 分割交差検証で線形に組み合わせた予測と、ベースライン単独を同じ手順で回帰し直したものとの差。",
             f"[ ] は graphic_id クラスタ・ブートストラップ（{n_bs} 回）の 95% 区間。", "",
             f"**判定基準（事前に決定）**：{PRIMARY[0]} + {PRIMARY[1]} の Δρ の区間下限 > 0 なら Go、点推定 > 0 で区間が 0 を含むなら弱い、点推定 ≤ 0 なら No-Go。", ""]
    verdict = None
    for name, sub in [("all", df), ("original", df[df.is_original]), ("ip", df[~df.is_original])]:
        est = stats(sub)
        ci = cluster_bootstrap(sub.reset_index(drop=True), stats, "graphic_id", n_bs, seed)
        fmt = lambda k: f"{est[k]:.3f} [{ci.loc[k, 'lo']:.3f}, {ci.loc[k, 'hi']:.3f}]"  # noqa: E731
        lines += [f"## {name}（n={len(sub)}）", "", "パネル単独の ρ：" + "、".join(f"{c} {fmt(f'rho|panel_{c}')}" for c in conds), "",
                  "| ベースライン | 単独 ρ | 単独 R² | 追加 | Δρ | ΔR² | ΔPrecision@20% |", "|---|---|---|---|---|---|---|"]
        for b in BASELINES:
            for c in [*conds, "both"]:
                lines.append(f"| {b} | {est[f'rho|{b}']:.3f} | {est[f'r2|{b}']:.3f} | {c} | {fmt(f'drho|{b}+{c}')} | {fmt(f'dr2|{b}+{c}')} | {fmt(f'dp20|{b}+{c}')} |")
        lines.append("")
        if name == "all":
            k = f"drho|{PRIMARY[0]}+{PRIMARY[1]}"
            lo, pt = ci.loc[k, "lo"], est[k]
            verdict = "Go" if lo > 0 else ("弱い" if pt > 0 else "No-Go")
    lines.insert(7, f"**判定：{verdict}**")
    lines.insert(8, "")
    path = REPO_ROOT / "reports" / "panel_stage1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
