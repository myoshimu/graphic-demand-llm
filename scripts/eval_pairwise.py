"""2 案比較の評価 → reports/pairwise.md（合格基準は decisions_log 2026-10-03（8）、結果を見る前に固定）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.bq import REPO_ROOT  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"


def correct(score: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """score > 0 なら first が上と予測。同点は 0.5 点。truth は first が多く売れたら 1。"""
    return np.where(score == 0, 0.5, ((score > 0) == (truth == 1)).astype(float))


def main(tag: str = "", period: str = "テスト期間（2025 年発売）") -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    n_bs, seed = int(settings["evaluation"]["bootstrap"]["n_resamples"]), int(settings["random_seed"])
    pairs = pd.read_parquet(PROC / "panel" / f"pairwise{tag}_pairs.parquet")
    resp = pd.read_parquet(PROC / "panel" / f"pairwise{tag}_responses.parquet")
    p = resp.groupby("pair").p_first.mean()  # 2 回の平均（片方が失敗なら残りの 1 回）
    pairs["p_pair"] = p.reindex(pairs.index).to_numpy()
    sc = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet").query("condition == 'full'")
    single = sc.set_index(["class_id", "color_id"]).intent_mean
    for side in ("first", "second"):
        pairs[f"{side}_single"] = [single.get((a, b), np.nan) for a, b in zip(pairs[f"{side}_class_id"], pairs[f"{side}_color_id"])]
    pairs["truth"] = (pairs.first_y_raw > pairs.second_y_raw).astype(int)
    pairs["s_pair"] = pairs.p_pair.fillna(0.5) - 0.5
    pairs["s_single"] = pairs.first_single - pairs.second_single
    pairs["s_B2"] = pairs.first_pred_B2_lgbm - pairs.second_pred_B2_lgbm
    pairs["s_B1"] = pairs.first_pred_B1_lgbm - pairs.second_pred_B1_lgbm

    # B2 と 2 案比較の組み合わせ：グループ単位の交差検証でロジスティック回帰（切片なし、順序に対称）
    pairs["s_combo"] = np.nan
    for st, g in pairs.groupby("set"):
        X, y = g[["s_B2", "s_pair"]].to_numpy(), g.truth.to_numpy()
        oof = np.empty(len(g))
        for tr, va in GroupKFold(5).split(X, groups=g.group):
            oof[va] = LogisticRegression(fit_intercept=False).fit(X[tr], y[tr]).decision_function(X[va])
        pairs.loc[g.index, "s_combo"] = oof

    methods = ["s_pair", "s_single", "s_B2", "s_B1", "s_combo"]
    if pairs["s_single"].isna().all():  # 1 商品ずつの評価は 2025 年分しか回していない
        methods.remove("s_single")
    for m in methods:
        pairs[f"c_{m}"] = correct(pairs[m].to_numpy(), pairs.truth.to_numpy())

    rng = np.random.default_rng(seed)
    lines = ["# 2 案比較：同じ IP 内のデザイン優劣をどこまで当てられるか", "",
             f"`scripts/eval_pairwise.py` が自動生成。{period}。正解率は「どちらがよく売れたか」を当てた割合（偶然なら 50%）。",
             f"[ ] は IP（オリジナルは発売月）単位のクラスタ・ブートストラップ（{n_bs} 回）の 95% 区間。", "",
             "**合格基準（事前に決定）**：IP の組で、2 案比較の正解率の区間下限 > 50% かつ（2 案比較 − B2）の区間下限 > 0 なら合格、"
             "前者のみなら部分合格、それ以外は不合格。", ""]
    verdict = None
    labels = {"s_pair": "2 案比較（Pro）", "s_single": "1 商品ずつの評価（段階 1、Pro full）", "s_B2": "B2_lgbm",
              "s_B1": "B1_lgbm", "s_combo": "B2 ＋ 2 案比較（交差検証で組み合わせ）"}
    for st, g in pairs.groupby("set"):
        groups = g.group.unique()
        idx = {k: np.where(g.group.to_numpy() == k)[0] for k in groups}
        C = g[[f"c_{m}" for m in methods]].to_numpy()
        est = C.mean(0)
        bs = []
        for _ in range(n_bs):
            ii = np.concatenate([idx[k] for k in rng.choice(groups, len(groups))])
            bs.append(C[ii].mean(0))
        bs = np.array(bs)
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        d = bs - bs[:, [methods.index("s_B2")]]
        dlo, dhi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
        dest = est - est[methods.index("s_B2")]
        n_fail = int(resp[resp.pair.isin(g.index)].parse_error.notna().sum())
        lines += [f"## {'IP 内' if st == 'ip' else 'オリジナル（同じ月）'}（{len(g)} 組、{len(groups)} グループ、パース失敗 {n_fail} 回）", "",
                  "| 方法 | 正解率 | B2 との差 |", "|---|---|---|"]
        for j, m in enumerate(methods):
            diff = "—" if m == "s_B2" else f"{dest[j]:+.3f} [{dlo[j]:+.3f}, {dhi[j]:+.3f}]"
            lines.append(f"| {labels[m]} | {est[j]:.3f} [{lo[j]:.3f}, {hi[j]:.3f}] | {diff} |")
        # 確信度が高い組だけの正解率
        conf = (g.p_pair - 0.5).abs().rank(method="first", pct=True)  # 確信度は離散値なので順位で切る
        for q in (0.5, 0.75):
            sel = conf > q
            lines.append(f"\n確信度の上位 {int((1 - q) * 100)}%（{int(sel.sum())} 組）の 2 案比較の正解率：{g.loc[sel, 'c_s_pair'].mean():.3f}")
        pos = resp[resp.pair.isin(g.index)]
        agree = pos.pivot(index="pair", columns="order", values="p_first").dropna()
        lines.append(f"\nA/B を入れ替えても同じ方を選んだ割合：{((agree.fs > 0.5) == (agree.sf > 0.5)).mean():.3f}")
        lines.append("")
        if st == "ip":
            j = methods.index("s_pair")
            verdict = "合格" if (lo[j] > 0.5 and dlo[j] > 0) else ("部分合格" if lo[j] > 0.5 else "不合格")
    lines.insert(6, f"**判定：{verdict}**")
    lines.insert(7, "")
    path = REPO_ROOT / "reports" / f"pairwise{tag}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
