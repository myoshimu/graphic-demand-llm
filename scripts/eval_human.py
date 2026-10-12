"""関係者による参考比較（2 案比較 30 組）の採点 → reports/human_eval.md（decisions_log 2026-10-10）。

入力：data/raw/human_eval/survey_result.csv（Google フォームの回答）と reports/human_eval/key.parquet（正解表）。
氏名は読み込んだ直後に捨て、回答時刻順の記号 R1〜R7 にする。匿名化した回答は data/processed/human_eval_responses.parquet。
同じ 30 組での LLM（Pro / Flash / Claude）と B2 も並べる。30 組では誤差幅が大きいので検定はしない。
分析の対象は data/raw/human_eval/included_raters.txt に書いた回答者だけ（売上データを確実に見ていない人、decisions_log 2026-10-10（3））。
氏名を含むのでこのファイルは data/raw/（git 管理外）に置く。7 名全員の結果は参考としてだけ併記する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_pairwise_robustness import load  # noqa: E402
from src.ingest.bq import REPO_ROOT  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "human_eval" / "survey_result.csv"
KEY = REPO_ROOT / "reports" / "human_eval" / "key.parquet"
INCLUDED = REPO_ROOT / "data" / "raw" / "human_eval" / "included_raters.txt"
MODELS = {"pro_v0": "Gemini Pro（主結果）", "flash_v0": "Gemini Flash", "claude46_v0": "Claude Sonnet 4.6", "B2": "B2"}


def load_responses() -> pd.DataFrame:
    """縦長（回答者 × 問）。氏名は使わない。"""
    raw = pd.read_csv(RAW).sort_values("タイムスタンプ").reset_index(drop=True)
    raw["rater"] = [f"R{i + 1}" for i in range(len(raw))]
    raw["group"] = np.where(raw["所属"].str.contains("retailer"), "retailer 社員", "研究室")
    names = [ln.strip() for ln in INCLUDED.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    missing = set(names) - set(raw["お名前"])
    if missing:
        raise ValueError(f"included_raters.txt の {len(missing)} 名が回答にありません")
    raw["included"] = raw["お名前"].isin(names)
    rows = []
    for _, r in raw.iterrows():
        for q in range(1, 31):
            rows.append({"rater": r["rater"], "group": r["group"], "included": bool(r["included"]), "q": q,
                         "choice": r[f"Q{q} 売れたのは"], "conf": int(r[f"Q{q} 確信度"])})
    return pd.DataFrame(rows)


def model_picks(key: pd.DataFrame) -> pd.DataFrame:
    """各問で各モデルが A を選んだか（1 / 0、同点は 0.5）。"""
    d = load()
    d26 = d[d["year"] == 2026].set_index("pair")
    out = key[["q"]].copy()
    for m in MODELS:
        picks = []
        for _, k in key.iterrows():
            r = d26.loc[k["pair"]]
            a_first = k["A_class_id"] == r["first_class_id"]
            s = r[m] if a_first else -r[m]
            picks.append(0.5 if s == 0 else float(s > 0))
        out[m] = picks
    return out


def fleiss_kappa(choice_a: pd.DataFrame) -> float:
    """choice_a：問 × 回答者の 0/1（A を選んだら 1）。2 カテゴリの Fleiss の κ。"""
    n = choice_a.shape[1]
    na = choice_a.sum(axis=1)
    counts = np.c_[na, n - na]
    p_i = (np.square(counts).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = np.square(p_j).sum()
    return float((p_i.mean() - pe) / (1 - pe))


def write_answer_key(key: pd.DataFrame, majority: np.ndarray, picks: pd.DataFrame) -> None:
    """研究責任者が自分で比べるための正解表 → reports/human_eval/answer_key.csv（Excel で開けるよう BOM 付き UTF-8）。

    売上の実数は載せず、どちらが多く売れたかだけを書く。問番号・A/B はフォームと同じ。
    商品は画像 URL で示す（商品 ID のままだと表計算ソフトが数値に変えて先頭の 0 が消えるため。URL なら崩れず画像も開ける）。
    """
    man = pd.read_parquet(REPO_ROOT / "data" / "cache" / "image_manifest.parquet").set_index(["class_id", "color_id"])["url"]
    pairs = pd.read_parquet(REPO_ROOT / "data" / "processed" / "panel" / "pairwise_all_pairs.parquet")
    p26 = pairs[pairs["year"] == 2026].set_index("pair")
    rows = []
    for i, k in key.reset_index(drop=True).iterrows():
        r = p26.loc[k["pair"]]
        a_first = k["A_class_id"] == r["first_class_id"]
        a, b = ("first", "second") if a_first else ("second", "first")
        row = {"問": int(k["q"]), "区分": "IP" if k["set"] == "ip" else "オリジナル",
               "IP 名": r["first_content_name"] if k["set"] == "ip" else "",
               "A の商品画像": man.loc[(r[f"{a}_class_id"], r[f"{a}_color_id"])],
               "B の商品画像": man.loc[(r[f"{b}_class_id"], r[f"{b}_color_id"])],
               "正解（多く売れた方）": k["truth"], "回答者の多数決": majority[i]}
        for m, lab in MODELS.items():
            v = picks.loc[k["q"], m]
            row[lab] = "同点" if v == 0.5 else ("A" if v == 1 else "B")
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPO_ROOT / "reports" / "human_eval" / "answer_key.csv", index=False, encoding="utf-8-sig")


def majority_vote(resp: pd.DataFrame, q_order: pd.Series) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    wide = resp.pivot(index="q", columns="rater", values="choice").reindex(q_order)
    vote_a = (wide == "A").sum(axis=1)
    return wide, vote_a, np.where(vote_a > wide.shape[1] / 2, "A", "B")


def main() -> None:
    key = pd.read_parquet(KEY)
    resp_all = load_responses().merge(key[["q", "set", "truth"]], on="q")
    resp_all["correct"] = (resp_all["choice"] == resp_all["truth"]).astype(float)
    resp_all.drop(columns=["truth"]).to_parquet(REPO_ROOT / "data" / "processed" / "human_eval_responses.parquet", index=False)
    resp = resp_all[resp_all["included"]]
    n = resp.rater.nunique()

    truth = key.set_index("q")["truth"]
    sets = key.set_index("q")["set"]
    wide, vote_a, majority = majority_vote(resp, key["q"])
    _, _, majority_all = majority_vote(resp_all, key["q"])
    picks = model_picks(key).set_index("q")
    write_answer_key(key, majority, picks)

    def acc(correct: pd.Series) -> str:
        c = correct.groupby(sets.reindex(correct.index)).mean()
        return f"{correct.mean():.1%} | {c.get('ip', np.nan):.1%} | {c.get('original', np.nan):.1%}"

    def mean_acc(r: pd.DataFrame) -> str:
        per_set = r.groupby(["rater", "set"])["correct"].mean().unstack()
        return f"{r.groupby('rater').correct.mean().mean():.1%} | {per_set['ip'].mean():.1%} | {per_set['original'].mean():.1%}"

    L = ["# 関係者による参考比較（2 案比較 30 組、2026 年発売）", "",
         "`scripts/eval_human.py` が自動生成。分析ルールは decisions_log 2026-10-10、対象者の絞り込みは同（3）。",
         f"対象は売上データを確実に見ていない {n} 名（{', '.join(sorted(resp.rater.unique()))}）。",
         "30 組では誤差幅が ±18pt 程度あるので検定はしない。IP 15 組・オリジナル 15 組。偶然なら 50%。", "",
         "| 回答者 | 区分 | 全体 | IP | オリジナル |", "|---|---|---:|---:|---:|"]
    for rater, g in resp.groupby("rater", sort=False):
        L.append(f"| {rater} | {g['group'].iloc[0]} | {acc(g.set_index('q')['correct'])} |")
    L.append(f"| 回答者の平均 | {n} 名 | {mean_acc(resp)} |")
    L.append(f"| 多数決 | {n} 名 | {acc(pd.Series(majority == truth.values, index=truth.index).astype(float))} |")
    for m, lab_m in MODELS.items():
        pa = picks[m]
        corr = np.where(pa == 0.5, 0.5, ((pa == 1) == (truth == "A")).astype(float))
        L.append(f"| {lab_m} | モデル | {acc(pd.Series(corr, index=truth.index))} |")
    L += ["", f"参考（売上の知識がある可能性のある回答者を含む 7 名全員、主の分析には使わない）：平均 {mean_acc(resp_all).split(' |')[0]}、"
          f"多数決 {(majority_all == truth.values).mean():.1%}。"]

    conf3 = resp[resp["conf"] == 3]
    L += ["", f"## その他（{n} 名）", "",
          f"- 評価者間一致（Fleiss の κ）：{fleiss_kappa((wide == 'A').astype(int)):.2f}",
          f"- 確信度 3（かなり自信あり）の回答：{len(conf3)} 件（全 {len(resp)} 件の {len(conf3) / len(resp):.0%}）、正解率 {conf3.correct.mean():.1%}",
          f"- 確信度 1 の回答の正解率：{resp[resp.conf == 1].correct.mean():.1%}（{(resp.conf == 1).sum()} 件）",
          f"- 多数決と Gemini Pro が同じ方を選んだ問：{int((majority == np.where(picks['pro_v0'] > 0.5, 'A', 'B')).sum())} / 30",
          f"- {n} 名全員が同じ方を選んだ問：{int(((vote_a == 0) | (vote_a == n)).sum())} / 30（うち正解 "
          f"{int((((vote_a == n) & (truth == 'A')) | ((vote_a == 0) & (truth == 'B'))).sum())}）", ""]
    path = REPO_ROOT / "reports" / "human_eval.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
