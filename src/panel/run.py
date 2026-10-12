"""パネルの実行（SKU × 条件 × ペルソナ）と集計（CLAUDE.md §5.2）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.panel import prompts
from src.panel.llm import BudgetExceeded, GeminiPanelClient
from src.panel.personas import Persona
from src.panel.stimuli import build_stimulus

log = structlog.get_logger()


def run_panel(client: GeminiPanelClient, skus: pd.DataFrame, conditions: list[str], personas: list[Persona], *,
              temperature: float, seed: int, variant: int = 0, workers: int = 8) -> pd.DataFrame:
    """1 呼び出し＝1 ペルソナ × 1 SKU × 1 条件。回答を長い形式で返す。予算超過時はそこまでの結果を返す。"""
    jobs: list[tuple[Any, ...]] = []
    for _, row in skus.iterrows():
        for cond in conditions:
            stim = build_stimulus(row, cond)
            for p in personas:
                jobs.append((stim, p))
    log.info("panel_start", model=client.model_key, n_calls=len(jobs), temperature=temperature, seed=seed, variant=variant)

    def one(stim: Any, p: Persona) -> dict[str, Any]:
        r = client.call(system=prompts.SYSTEM, user_text=prompts.build_user_text(p.describe_ja(), stim.text, variant),
                        image_path=stim.image_path, image_sha256=stim.image_sha256, is_original=stim.is_original,
                        temperature=temperature, seed=seed)
        out = {"class_id": stim.class_id, "color_id": stim.color_id, "condition": stim.condition,
               "persona_id": p.persona_id, "model_key": client.model_key, "temperature": temperature, "seed": seed,
               "variant": variant, "cache_hit": r.cache_hit, "usd_cost": r.usd_cost, "parse_error": r.parse_error}
        out.update(r.response.model_dump() if r.response else {})
        return out

    rows: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, s, p) for s, p in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    rows.append(f.result())
                except BudgetExceeded as e:
                    log.error("budget_exceeded", msg=str(e))
                    for g in futs:
                        g.cancel()
                    break
                if i % 500 == 0:
                    log.info("panel_progress", model=client.model_key, done=i, total=len(jobs),
                             spent_usd=round(client.spent_usd, 2))
    finally:
        client.flush()
    return pd.DataFrame(rows)


def aggregate(responses: pd.DataFrame) -> pd.DataFrame:
    """SKU × 条件 × モデル（× 温度・seed・言い換え）ごとの mean / sd / top-box 比率。"""
    keys = ["class_id", "color_id", "condition", "model_key", "temperature", "seed", "variant"]
    ok = responses[responses["parse_error"].isna()]
    g = ok.groupby(keys)
    agg = g.agg(
        n=("purchase_intent", "size"),
        intent_mean=("purchase_intent", "mean"), intent_sd=("purchase_intent", "std"),
        top_box=("purchase_intent", lambda x: float(np.mean(x == 5))),
        top2_box=("purchase_intent", lambda x: float(np.mean(x >= 4))),
        prob_mean=("purchase_prob", "mean"), prob_sd=("purchase_prob", "std"),
        wtp_median=("would_pay_jpy", "median"),
    ).reset_index()
    n_all = responses.groupby(keys).size().rename("n_calls").reset_index()
    return agg.merge(n_all, on=keys, how="right")
