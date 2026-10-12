"""2 案比較の理由文の記述的分析（主結果 Pro・専門家プロンプト、2025＋2026 年）→ reports/pairwise_reasons.md。

理由文をキーワード辞書で論点に振り分け、論点ごとの出現率と、その論点を挙げたときの正解率を出す。
辞書は理由文を数十件読んで作った粗いもの（事後・記述的。仮説検定には使わない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.bq import REPO_ROOT  # noqa: E402

PROC = REPO_ROOT / "data" / "processed" / "panel"
THEMES = {
    "キャラ・作品の人気": ["人気", "知名度", "主人公", "メインキャラ", "看板", "代表", "名シーン", "象徴"],
    "ファン訴求": ["ファン", "刺さ", "心を掴"],
    "日常での着やすさ": ["日常", "着こな", "普段", "使いやす", "合わせやす", "汎用", "幅広い"],
    "色・ボディ": ["色", "カラー", "ブラック", "ホワイト", "黒", "白"],
    "配置・構図・ギミック": ["配置", "構図", "ポケット", "ギミック", "バック", "背面", "ワンポイント", "大胆", "全面"],
    "おしゃれさ・洗練": ["おしゃれ", "洗練", "スタイリッシュ", "シンプル", "上品", "大人"],
    "インパクト・目を引く": ["インパクト", "目を引", "目立", "存在感", "印象的"],
    "かわいさ・ポップ": ["かわい", "可愛", "ポップ", "キュート", "ゆる"],
    "文字・ロゴ": ["ロゴ", "文字", "タイポ", "テキスト"],
    "価格": ["価格", "安", "円", "手頃", "値"],
}


def main() -> None:
    pairs = pd.read_parquet(PROC / "pairwise_all_pairs.parquet").set_index("pair_id")
    rs = []
    for tag, year in (("", 2025), ("_confirm2026", 2026)):
        r = pd.read_parquet(PROC / f"pairwise{tag}_responses.parquet")
        rs.append(r.assign(pair_id=f"{year}_" + r["pair"].astype(str)))
    r = pd.concat(rs).dropna(subset=["p_first"]).join(pairs[["set", "first_y_raw", "second_y_raw"]], on="pair_id")
    r = r[r.p_first != 0.5]
    # 回答 1 件ごとに「その回答が正しかったか」を判定（order が swap なら画像の並びが逆）
    r["correct"] = (r.p_first > 0.5) == (r.first_y_raw > r.second_y_raw)
    for th, kws in THEMES.items():
        r[th] = r.reason.fillna("").str.contains("|".join(kws))
    L = ["# 2 案比較の理由文の記述的分析", "",
         "`scripts/analyze_pairwise_reasons.py` が自動生成。対象は主結果（Pro・専門家プロンプト）の回答 1 件ずつ（A/B 両順を別々に数える）。",
         "論点はキーワード辞書による粗い振り分けで、事後の記述的分析。1 つの理由文が複数の論点に入ることがある。", ""]
    for st, title in (("ip", "IP 内"), ("original", "オリジナル")):
        g = r[r["set"] == st]
        L += [f"## {title}（回答 {len(g)} 件、全体の正解率 {g.correct.mean():.3f}）", "",
              "| 論点 | 出現率 | その論点を挙げた回答の正解率 | 挙げなかった回答の正解率 |", "|---|---:|---:|---:|"]
        rows = sorted(THEMES, key=lambda t: -g[t].mean())
        for th in rows:
            L.append(f"| {th} | {g[th].mean():.1%} | {g.loc[g[th], 'correct'].mean():.3f} | {g.loc[~g[th], 'correct'].mean():.3f} |")
        L.append("")
    path = REPO_ROOT / "reports" / "pairwise_reasons.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
