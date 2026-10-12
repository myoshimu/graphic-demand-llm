"""2 案比較の実行（decisions_log 2026-10-03（8））→ data/processed/panel/pairwise_responses.parquet。

テスト期間（2025 年発売）の母集団。主：同じ IP 内の全組。副：同じ月発売のオリジナル同士から無作為 1,000 組。
Pro、温度 0、seed 1、画像 high、ペルソナなし、A/B 入れ替えで 2 回。上限 $100（1 回約 $0.017）。IP の組から先に回す。
キャッシュがあるので、途中で止まっても再実行すれば続きから回る。
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.bq import REPO_ROOT  # noqa: E402
from src.panel.llm import BudgetExceeded, GeminiPanelClient  # noqa: E402
from src.panel.pairwise import SYSTEM, PairResponse, build_pair_text, make_pairs, p_choose_a  # noqa: E402
from src.panel.stimuli import load_sku_frame  # noqa: E402

log = structlog.get_logger()

PROC = REPO_ROOT / "data" / "processed"
MODEL_KEY = "api_a"
BUDGET_USD = 100.0
N_ORIGINAL_PAIRS = 1000
SEED = 20261003
MAX_TOKENS = 4096  # 2 案比較では Pro が budget 0 でも長く考える（試行で最大 1,600 トークン）。2048 では打ち切りの恐れ


def load_pairs(pred_file: str = "baseline_predictions.parquet") -> pd.DataFrame:
    base = pd.read_parquet(PROC / pred_file)[
        ["class_id", "color_id", "graphic_id", "is_original", "y_raw", "pred_B2_lgbm", "pred_B1_lgbm"]]
    skus = pd.read_parquet(PROC / "skus.parquet").drop_duplicates(["class_id", "color_id"])[
        ["class_id", "color_id", "content_name"]]
    frame = load_sku_frame()[["class_id", "color_id", "release_date", "price_list", "image_path", "image_sha256"]]
    d = base.merge(skus, on=["class_id", "color_id"]).merge(frame, on=["class_id", "color_id"])
    d["release_month"] = pd.to_datetime(d["release_date"]).dt.strftime("%Y-%m")
    ip = make_pairs(d[~d["is_original"]], "content_name").assign(set="ip")
    og = make_pairs(d[d["is_original"]], "release_month", N_ORIGINAL_PAIRS, SEED).assign(set="original")
    return pd.concat([ip, og], ignore_index=True)


def main(tag: str = "", pred_file: str = "baseline_predictions.parquet", budget_usd: float = BUDGET_USD) -> None:
    """tag は出力ファイル名の接尾辞（確認用 2026 年は "_confirm2026"、scripts/run_confirm.py から呼ぶ）。"""
    pairs = load_pairs(pred_file)
    client = GeminiPanelClient(MODEL_KEY, budget_usd=budget_usd, media_resolution="high", max_tokens=MAX_TOKENS)
    jobs = []
    for i, r in pairs.iterrows():
        f = {k[6:]: v for k, v in r.items() if k.startswith("first_")}
        s = {k[7:]: v for k, v in r.items() if k.startswith("second_")}
        for order, (a, b) in (("fs", (f, s)), ("sf", (s, f))):
            jobs.append((i, order, a, b))

    def one(job):
        i, order, a, b = job
        res = client.call(system=SYSTEM, user_text=build_pair_text(pd.Series(a), pd.Series(b)),
                          images=[(a["image_path"], a["image_sha256"]), (b["image_path"], b["image_sha256"])],
                          is_original=bool(a["is_original"]), temperature=0.0, seed=1, response_model=PairResponse)
        p = p_choose_a(res.response)
        return {"pair": i, "order": order, "p_first": p if order == "fs" else 1 - p,
                "parse_error": res.parse_error, "reason": res.response.reason if res.response else None}

    out = []
    try:
        with ThreadPoolExecutor(16) as ex:
            for n, r in enumerate(ex.map(one, jobs), 1):
                out.append(r)
                if n % 500 == 0:
                    log.info("pairwise_progress", done=n, total=len(jobs), spent_usd=round(client.spent_usd, 2))
    except BudgetExceeded as e:
        log.warning("budget_exceeded", msg=str(e))
    finally:
        client.flush()
    resp = pd.DataFrame(out)
    resp.to_parquet(PROC / "panel" / f"pairwise{tag}_responses.parquet", index=False)
    pairs.drop(columns=[c for c in pairs.columns if "image" in c]).to_parquet(PROC / "panel" / f"pairwise{tag}_pairs.parquet", index=False)
    log.info("pairwise_done", n_pairs=len(pairs), n_calls=len(resp), spent_usd=round(client.spent_usd, 2))


if __name__ == "__main__":
    main()
