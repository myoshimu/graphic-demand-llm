"""パネルのプロンプト（CLAUDE.md §5.2）。言い換え 3 種（variant 0–2）。

文面を変えるとキャッシュキー（prompt_sha256）が変わり、再呼び出しになる。変更時は decisions_log に記録する。
"""

from __future__ import annotations

SYSTEM = (
    "あなたはアンケートに答える一般の消費者です。以下の人物になりきって、"
    "retailer（グラフィック T シャツの専門店）の新商品について正直に答えてください。"
    "回答は指定された JSON だけを出力してください。"
)

QUESTION = [
    "この T シャツを見て、あなたはどのくらい買いたいと思いますか。",
    "店頭やオンラインストアでこの T シャツを見かけたとします。購入を検討しますか。",
    "この T シャツがいま発売されたら、あなたは買うと思いますか。",
]

RESPONSE_SPEC = (
    "次の形式の JSON で答えてください。\n"
    '{"purchase_intent": 1〜5 の整数（1=まったく買いたくない, 5=ぜひ買いたい）, '
    '"purchase_prob": 0〜1 の小数（この商品を買う確率）, '
    '"would_pay_jpy": 整数（この商品に払ってもよい上限の金額、円）, '
    '"reason": "30 字以内の理由"}'
)


def build_user_text(persona_text: str, stimulus_text: str, variant: int) -> str:
    parts = [f"【あなたについて】\n{persona_text}", f"【商品】\n{stimulus_text}", QUESTION[variant], RESPONSE_SPEC]
    return "\n\n".join(parts)
