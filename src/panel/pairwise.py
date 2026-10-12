"""2 案比較：同じ IP（または同じ月のオリジナル）の T シャツ 2 枚のどちらが売れるかを聞く（decisions_log 2026-10-03（8））。

1 組につき A/B の順序を入れ替えて 2 回聞き、位置の偏りを打ち消す。
p_first = 「先に挙げた方（first）が多く売れる」確率の推定 = 2 回の平均。
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

SYSTEM = (
    "あなたは retailer（グラフィック T シャツの専門店）で長年商品企画を担当してきたベテランです。"
    "デザインやイラストの配置、ボディ色を見て、どちらの商品がよく売れるかを判断してください。"
    "回答は指定された JSON だけを出力してください。"
)

RESPONSE_SPEC = (
    "次の形式の JSON で答えてください。\n"
    '{"choice": "A" または "B"（よく売れると思う方）, '
    '"confidence": 0.5〜1 の小数（その判断の確からしさ。0.5=五分五分）, '
    '"reason": "40 字以内の理由"}'
)


class PairResponse(BaseModel):
    choice: Literal["A", "B"]
    confidence: float = Field(ge=0.5, le=1)
    reason: str

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, v: object) -> object:
        # Pro はまれに 1 要素の配列で包んで返す
        return v[0] if isinstance(v, list) and len(v) == 1 else v


# ---- 頑健性用の言い換え（decisions_log 2026-10-04（13）。主結果は上の SYSTEM / build_pair_text のまま固定）----
SYSTEM_V1 = (
    "あなたはアパレル小売 retailer のバイヤーです。グラフィック T シャツの売れ行きを予想する仕事をしています。"
    "2 枚の商品画像を比べ、どちらが多く売れたかを予想してください。回答は指定された JSON だけを出力してください。"
)

SYSTEM_CONSUMER = (
    "あなたはアンケートに答える一般の消費者です。以下の人物になりきって、"
    "retailer（グラフィック T シャツの専門店）の T シャツについて正直に答えてください。"
    "回答は指定された JSON だけを出力してください。"
)

RESPONSE_SPEC_CONSUMER = (
    "次の形式の JSON で答えてください。\n"
    '{"choice": "A" または "B"（買う方）, '
    '"confidence": 0.5〜1 の小数（その選択の強さ。0.5=どちらでもよい）, '
    '"reason": "40 字以内の理由"}'
)


def build_pair_text_v1(a: pd.Series, b: pd.Series) -> str:
    lines = [
        context_line(a),
        f"A：{int(a['price_list']):,} 円（税込）／B：{int(b['price_list']):,} 円（税込）",
        "発売から 8 週間、オンラインストアで多く売れたのは A と B のどちらだと予想しますか。",
        RESPONSE_SPEC,
    ]
    return "\n\n".join(lines)


def build_pair_text_consumer(a: pd.Series, b: pd.Series, persona_text: str) -> str:
    lines = [
        f"【あなたについて】\n{persona_text}",
        context_line(a),
        f"画像 A の価格：{int(a['price_list']):,} 円（税込）",
        f"画像 B の価格：{int(b['price_list']):,} 円（税込）",
        "どちらか 1 枚だけ買うとしたら、A と B のどちらを買いますか。",
        RESPONSE_SPEC_CONSUMER,
    ]
    return "\n\n".join(lines)


def context_line(row: pd.Series) -> str:
    if row["is_original"]:
        return "どちらも retailer オリジナルデザインで、同じ月に発売された T シャツです。"
    return f"どちらも「{row['content_name']}」とのコラボ（ライセンス）T シャツです。"


def build_pair_text(a: pd.Series, b: pd.Series) -> str:
    lines = [
        context_line(a),
        f"画像 A の価格：{int(a['price_list']):,} 円（税込）",
        f"画像 B の価格：{int(b['price_list']):,} 円（税込）",
        "発売後 8 週間のオンラインストアでの販売数が多いのは、A と B のどちらだと思いますか。",
        RESPONSE_SPEC,
    ]
    return "\n\n".join(lines)


def make_pairs(df: pd.DataFrame, group_col: str, max_pairs: int | None = None, seed: int = 0) -> pd.DataFrame:
    """group_col が同じ SKU 同士の全組（売上が同値の組と同一 graphic_id の組は除く）。

    max_pairs を超える場合は seed 付きで無作為に間引く。返す列：first_*, second_*, group。
    """
    rows = []
    for key, g in df.groupby(group_col):
        for a, b in itertools.combinations(g.to_dict("records"), 2):
            if a["y_raw"] == b["y_raw"] or a["graphic_id"] == b["graphic_id"]:
                continue
            rows.append({"group": str(key), **{f"first_{k}": v for k, v in a.items()},
                         **{f"second_{k}": v for k, v in b.items()}})
    pairs = pd.DataFrame(rows)
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs.sample(max_pairs, random_state=seed)
    return pairs.reset_index(drop=True)


def p_choose_a(resp: PairResponse | None) -> float:
    if resp is None:
        return np.nan
    return resp.confidence if resp.choice == "A" else 1 - resp.confidence
