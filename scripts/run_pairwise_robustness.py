"""2 案比較の頑健性（decisions_log 2026-10-04（13））→ data/processed/panel/pairwise_rob_{name}.parquet。

主結果（Pro・専門家プロンプト v0）は固定したまま、次の 3 つを 2025＋2026 年の組で回す。
- flash_v0：Flash（3.5、thinking MINIMAL）で同じプロンプト。IP とオリジナルの全組、A/B 入れ替えで 2 回
- pro_v1：Pro で言い換えプロンプト（バイヤーとして予想）。IP の全組、A/B 入れ替えで 2 回
- flash_consumer：Flash で消費者ペルソナ 10 人に「どちらを買うか」。IP の全組。ペルソナごとに A/B の順を交互にする
- claude46_v0：Claude Sonnet 4.6 で主結果と同じプロンプト。IP とオリジナルの全組、A/B 入れ替えで 2 回
  （事後に追加した系統の異なるモデルでの再現、基準は decisions_log 2026-10-07（2））
組は sets の順（IP が先）に回すので、予算上限で止まっても IP は揃う。
キャッシュがあるので再実行しても再課金されない。
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.bq import REPO_ROOT  # noqa: E402
from src.panel import pairwise as pw  # noqa: E402
from src.panel.llm import BudgetExceeded, make_client  # noqa: E402
from src.panel.personas import sample_personas  # noqa: E402
from src.panel.stimuli import load_sku_frame  # noqa: E402

log = structlog.get_logger()
PROC = REPO_ROOT / "data" / "processed" / "panel"
MAX_TOKENS = 4096
N_PERSONAS = 10

RUNS = {
    "flash_v0": {"model": "api_b", "sets": ("ip", "original"), "budget": 40.0},
    "pro_v1": {"model": "api_a", "sets": ("ip",), "budget": 60.0},
    "flash_consumer": {"model": "api_b", "sets": ("ip",), "budget": 70.0},
    "claude46_v0": {"model": "api_c", "sets": ("ip", "original"), "budget": 100.0},
}


def load_all_pairs() -> pd.DataFrame:
    frames = []
    for tag, year in (("", 2025), ("_confirm2026", 2026)):
        p = pd.read_parquet(PROC / f"pairwise{tag}_pairs.parquet").reset_index(names="pair")
        frames.append(p.assign(year=year, pair_id=lambda d, y=year: f"{y}_" + d["pair"].astype(str)))
    pairs = pd.concat(frames, ignore_index=True)
    img = load_sku_frame()[["class_id", "color_id", "image_path", "image_sha256"]]
    for side in ("first", "second"):
        pairs = pairs.merge(img.add_prefix(f"{side}_"), on=[f"{side}_class_id", f"{side}_color_id"], how="left")
    return pairs


def jobs_for(name: str, pairs: pd.DataFrame) -> list[dict]:
    spec, jobs = RUNS[name], []
    personas = sample_personas(N_PERSONAS)
    ordered = pd.concat([pairs[pairs["set"] == st] for st in spec["sets"]])
    for _, r in ordered.iterrows():
        f = pd.Series({k[6:]: v for k, v in r.items() if k.startswith("first_")})
        s = pd.Series({k[7:]: v for k, v in r.items() if k.startswith("second_")})
        if name == "flash_consumer":
            for per in personas:
                fs = per.persona_id % 2 == 0
                a, b = (f, s) if fs else (s, f)
                jobs.append({"pair_id": r["pair_id"], "order": "fs" if fs else "sf", "persona_id": per.persona_id,
                             "system": pw.SYSTEM_CONSUMER, "text": pw.build_pair_text_consumer(a, b, per.describe_ja()), "a": a, "b": b})
        else:
            for order, (a, b) in (("fs", (f, s)), ("sf", (s, f))):
                system, text = ((pw.SYSTEM, pw.build_pair_text(a, b)) if name in ("flash_v0", "claude46_v0")
                                else (pw.SYSTEM_V1, pw.build_pair_text_v1(a, b)))
                jobs.append({"pair_id": r["pair_id"], "order": order, "persona_id": -1, "system": system, "text": text, "a": a, "b": b})
    return jobs


def run(name: str, pairs: pd.DataFrame) -> None:
    spec = RUNS[name]
    client = make_client(spec["model"], budget_usd=spec["budget"], media_resolution="high", max_tokens=MAX_TOKENS)
    jobs = jobs_for(name, pairs)

    def one(j: dict) -> dict:
        a, b = j["a"], j["b"]
        res = client.call(system=j["system"], user_text=j["text"],
                          images=[(a["image_path"], a["image_sha256"]), (b["image_path"], b["image_sha256"])],
                          is_original=bool(a["is_original"]), temperature=0.0, seed=1, response_model=pw.PairResponse)
        p = pw.p_choose_a(res.response)
        return {"pair_id": j["pair_id"], "order": j["order"], "persona_id": j["persona_id"],
                "p_first": p if j["order"] == "fs" else 1 - p, "parse_error": res.parse_error}

    out = []
    try:
        with ThreadPoolExecutor(16) as ex:
            for n, r in enumerate(ex.map(one, jobs), 1):
                out.append(r)
                if n % 1000 == 0:
                    log.info("rob_progress", run=name, done=n, total=len(jobs), spent_usd=round(client.spent_usd, 2))
    except BudgetExceeded as e:
        log.warning("budget_exceeded", run=name, msg=str(e))
    finally:
        client.flush()
    pd.DataFrame(out).assign(run=name, model_key=spec["model"]).to_parquet(PROC / f"pairwise_rob_{name}.parquet", index=False)
    log.info("rob_done", run=name, n_calls=len(out), spent_usd=round(client.spent_usd, 2))


def main() -> None:
    pairs = load_all_pairs()
    pairs.drop(columns=[c for c in pairs.columns if "image" in c]).to_parquet(PROC / "pairwise_all_pairs.parquet", index=False)
    for name in sys.argv[1:] or RUNS:
        run(name, pairs)


if __name__ == "__main__":
    main()
