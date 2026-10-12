"""2 案比較の頑健性・感度分析の評価 → reports/pairwise_robustness.md（decisions_log 2026-10-04（13））。

2025＋2026 年の組をあわせ、IP×年（オリジナルは発売月）をクラスタにしたブートストラップで区間を出す。
主結果（pro_v0）は事前登録どおりの値で、ここでの他の方法はすべて事後の感度分析。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.bq import REPO_ROOT  # noqa: E402

PROC = REPO_ROOT / "data" / "processed" / "panel"
LABELS = {
    "pro_v0": "Pro・専門家プロンプト（主結果）",
    "pro_v1": "Pro・言い換え（バイヤー）",
    "pro_v0v1": "Pro・v0 と v1 の平均",
    "flash_v0": "Flash・専門家プロンプト",
    "flash_consumer": "Flash・消費者ペルソナ 10 人の多数決",
    "claude46_v0": "Claude Sonnet 4.6・専門家プロンプト（事後追加）",
    "B2": "B2_lgbm",
    "B1": "B1_lgbm",
}


def correct(score: pd.Series, truth: pd.Series) -> pd.Series:
    """score > 0 なら first が上と予測。同点は 0.5 点。score が NaN（未実施）は NaN。"""
    out = np.where(score == 0, 0.5, ((score > 0) == (truth == 1)).astype(float))
    return pd.Series(np.where(score.isna(), np.nan, out), index=score.index)


def load() -> pd.DataFrame:
    pairs = pd.read_parquet(PROC / "pairwise_all_pairs.parquet").set_index("pair_id")
    v0 = []
    for tag, year in (("", 2025), ("_confirm2026", 2026)):
        r = pd.read_parquet(PROC / f"pairwise{tag}_responses.parquet")
        v0.append(r.assign(pair_id=f"{year}_" + r["pair"].astype(str)))
    scores = {"pro_v0": pd.concat(v0).groupby("pair_id").p_first.mean()}
    for name in ("flash_v0", "pro_v1", "flash_consumer", "claude46_v0"):
        path = PROC / f"pairwise_rob_{name}.parquet"
        if path.exists():
            scores[name] = pd.read_parquet(path).groupby("pair_id").p_first.mean()
    d = pairs.copy()
    for k, v in scores.items():
        d[k] = v.reindex(d.index) - 0.5
    if "pro_v1" in d:
        d["pro_v0v1"] = d[["pro_v0", "pro_v1"]].mean(axis=1, skipna=False)
    d["B2"] = d.first_pred_B2_lgbm - d.second_pred_B2_lgbm
    d["B1"] = d.first_pred_B1_lgbm - d.second_pred_B1_lgbm
    d["truth"] = (d.first_y_raw > d.second_y_raw).astype(int)
    d["cluster"] = d["group"].astype(str) + "_" + d["year"].astype(str)
    return d


def boot(g: pd.DataFrame, cols: list[str], n_bs: int, rng: np.random.Generator) -> pd.DataFrame:
    C = g[[f"c_{c}" for c in cols]].to_numpy()
    cl = g["cluster"].to_numpy()
    keys = np.unique(cl)
    idx = {k: np.where(cl == k)[0] for k in keys}
    est = np.nanmean(C, 0)
    bs = np.array([np.nanmean(C[np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])], 0) for _ in range(n_bs)])
    j = cols.index("B2")
    return pd.DataFrame({
        "n": (~np.isnan(C)).sum(0), "acc": est, "lo": np.nanpercentile(bs, 2.5, 0), "hi": np.nanpercentile(bs, 97.5, 0),
        "d": est - est[j], "dlo": np.nanpercentile(bs - bs[:, [j]], 2.5, 0), "dhi": np.nanpercentile(bs - bs[:, [j]], 97.5, 0),
    }, index=cols)


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    n_bs, seed = int(settings["evaluation"]["bootstrap"]["n_resamples"]), int(settings["random_seed"])
    d = load()
    methods = [m for m in LABELS if m in d]
    for m in methods:
        d[f"c_{m}"] = correct(d[m], d["truth"])
    rng = np.random.default_rng(seed)
    L = ["# 2 案比較の頑健性・感度分析（2025＋2026 年）", "",
         "`scripts/eval_pairwise_robustness.py` が自動生成。正解率は偶然なら 50%。"
         f"[ ] は IP×年（オリジナルは発売月）クラスタ・ブートストラップ（{n_bs} 回）の 95% 区間。",
         "主結果（Pro・専門家プロンプト）以外はすべて事後の感度分析。B2 との差は、その方法を回した組だけで比べている。", ""]
    for st, title in (("ip", "IP 内"), ("original", "オリジナル（同じ月）")):
        L += [f"## {title}", "", "| 方法 | 組数 | 正解率 | B2 との差 |", "|---|---:|---|---|"]
        g = d[d["set"] == st]
        for m in methods:
            if m == "B2" or g[m].isna().all():
                continue
            sub = g[g[m].notna()]
            t = boot(sub, [m, "B2"], n_bs, rng)
            L.append(f"| {LABELS[m]} | {int(t.at[m, 'n'])} | {t.at[m, 'acc']:.3f} [{t.at[m, 'lo']:.3f}, {t.at[m, 'hi']:.3f}] | "
                     f"{t.at[m, 'd']:+.3f} [{t.at[m, 'dlo']:+.3f}, {t.at[m, 'dhi']:+.3f}] |")
        t = boot(g, ["B2"], n_bs, rng)
        L.append(f"| B2_lgbm | {int(t.at['B2', 'n'])} | {t.at['B2', 'acc']:.3f} [{t.at['B2', 'lo']:.3f}, {t.at['B2', 'hi']:.3f}] | — |")
        # 確信度で絞ったときの正解率（主結果）と、確信度が低い組は B2 に任せる切り替え
        L += ["", "**確信度で絞った場合（Pro・専門家プロンプト）**", "",
              "| 判定する組の割合 | 2 案比較の正解率 | 残りを B2 に任せた全体の正解率 |", "|---:|---:|---:|"]
        conf = g["pro_v0"].abs().rank(method="first", pct=True)
        for cov in (1.0, 0.75, 0.5, 0.25):
            sel = conf > 1 - cov
            mixed = np.where(sel, g["c_pro_v0"], g["c_B2"]).mean()
            L.append(f"| {cov:.0%}（{int(sel.sum())} 組） | {g.loc[sel, 'c_pro_v0'].mean():.3f} | {mixed:.3f} |")
        L += ["", "**年別（主結果と B2）**", ""]
        for y, gy in g.groupby("year"):
            L.append(f"- {y} 年：Pro {gy.c_pro_v0.mean():.3f} / B2 {gy.c_B2.mean():.3f}（{len(gy)} 組）")
        L.append("")
    if "pro_v1" in d:
        ip = d[(d["set"] == "ip") & d["pro_v1"].notna()]
        L += ["## プロンプト間・モデル間の一致", "",
              f"- Pro v0 と v1 で同じ方を選んだ割合（IP）：{((ip.pro_v0 > 0) == (ip.pro_v1 > 0)).mean():.3f}"]
        for m in ("flash_v0", "flash_consumer", "claude46_v0"):
            if m in d:
                s = d[(d["set"] == "ip") & d[m].notna()]
                L.append(f"- Pro v0 と {LABELS[m]} で同じ方を選んだ割合（IP）：{((s.pro_v0 > 0) == (s[m] > 0)).mean():.3f}")
        L.append("")
    if "claude46_v0" in d:
        # 汚染チェック（decisions_log 2026-10-07（2））：両方の商品が学習データのカットオフ（2026-01）後に発売された組
        after = (pd.to_datetime(d.first_release_date) >= "2026-02-01") & (pd.to_datetime(d.second_release_date) >= "2026-02-01")
        L += ["## Claude の学習データのカットオフ後（両方とも 2026-02-01 以降に発売）", "",
              "Sonnet 4.6 の学習データは 2026-01 まで。この組は学習に含まれ得ない。", "",
              "| 区分 | 組数 | Claude | Pro v0 | B2 との差（Claude） |", "|---|---:|---|---:|---|"]
        for st, title in (("ip", "IP 内"), ("original", "オリジナル")):
            g = d[after & (d["set"] == st) & d["claude46_v0"].notna()]
            t = boot(g, ["claude46_v0", "B2"], n_bs, rng)
            L.append(f"| {title} | {len(g)} | {t.at['claude46_v0', 'acc']:.3f} [{t.at['claude46_v0', 'lo']:.3f}, {t.at['claude46_v0', 'hi']:.3f}] | "
                     f"{g.c_pro_v0.mean():.3f} | {t.at['claude46_v0', 'd']:+.3f} [{t.at['claude46_v0', 'dlo']:+.3f}, {t.at['claude46_v0', 'dhi']:+.3f}] |")
        L.append("")
    path = REPO_ROOT / "reports" / "pairwise_robustness.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
