"""合成パネルのパイロット（docs/pilot_plan.md §3）→ reports/panel_pilot.md。

学習期間（2023–24 年発売）の母集団から 100 SKU（IP / オリジナル 50 ずつ、主アウトカムの 5 分位で層化）。
条件 full / image_masked_text × ペルソナ 50 × 2 モデル。温度 0、seed 1、言い換え 0、thinking なし、画像 high。
Flash は 20 SKU だけ画像 medium も回す。キャッシュがあるので再実行しても再課金されない。

環境変数 PILOT_SMOKE=1 で 2 SKU × 2 ペルソナだけ回す（動作確認用）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import spearman  # noqa: E402
from src.ingest.bq import REPO_ROOT, load_settings, require  # noqa: E402
from src.panel.llm import GeminiPanelClient  # noqa: E402
from src.panel.personas import sample_personas  # noqa: E402
from src.panel.run import aggregate, run_panel  # noqa: E402
from src.panel.stimuli import load_sku_frame  # noqa: E402

log = structlog.get_logger()

N_PER_GROUP = 50
CONDITIONS = ["full", "image_masked_text"]
BUDGET = {"api_a": 60.0, "api_b": 35.0}  # ＋ Flash medium 比較に $5。合計上限 $100（pilot_plan §3）。オンライン単価で計上
N_MEDIUM = 20
SEED = 20261003


def pick_skus(settings: dict) -> pd.DataFrame:
    proc = REPO_ROOT / require(settings, "paths.processed")
    pop = pd.read_parquet(proc / "population.parquet").query("in_population")
    y_col = require(settings, "outcomes.primary")
    out = pd.read_parquet(proc / "outcomes.parquet")[["class_id", y_col]]
    df = load_sku_frame().merge(pop[["class_id", "color_id"]], on=["class_id", "color_id"]).merge(out, on="class_id")
    train_years = require(settings, "evaluation.split.train")
    df = df[df["release_date"].dt.year.isin(train_years) & df["image_path"].notna()].copy()
    rng = np.random.default_rng(SEED)
    picks = []
    for is_orig, g in df.groupby("is_original"):
        g = g.assign(q=pd.qcut(g[y_col].rank(method="first"), 5, labels=False))
        for _, gq in g.groupby("q"):
            picks.append(gq.sample(N_PER_GROUP // 5, random_state=int(rng.integers(1 << 31))))
    return pd.concat(picks).rename(columns={y_col: "y"}).reset_index(drop=True)


def main() -> None:
    settings = load_settings()
    run_start = pd.Timestamp.now(tz="UTC").isoformat()
    smoke = os.environ.get("PILOT_SMOKE") == "1"
    skus = pick_skus(settings)
    personas = sample_personas(50)
    if smoke:
        skus = skus.groupby("is_original").head(1)
        personas = personas[:2]
    out_dir = REPO_ROOT / require(settings, "paths.processed") / "panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for key in ["api_b", "api_a"]:
        client = GeminiPanelClient(key, budget_usd=BUDGET[key], media_resolution="high")
        frames.append(run_panel(client, skus, CONDITIONS, personas, temperature=0.0, seed=1, workers=16).assign(media_resolution="high"))
        if key == "api_b":
            med = GeminiPanelClient(key, budget_usd=5.0, media_resolution="medium")
            sub = skus.groupby("is_original").head(N_MEDIUM // 2)
            frames.append(run_panel(med, sub, CONDITIONS, personas, temperature=0.0, seed=1, workers=16).assign(media_resolution="medium"))
    resp = pd.concat(frames, ignore_index=True)
    suffix = "_smoke" if smoke else ""
    resp.to_parquet(out_dir / f"pilot_responses{suffix}.parquet", index=False)
    skus.to_parquet(out_dir / f"pilot_skus{suffix}.parquet", index=False)
    write_report(resp, skus, smoke, run_start)


def write_report(resp: pd.DataFrame, skus: pd.DataFrame, smoke: bool, run_start: str) -> None:
    from src.panel.llm import load_models

    led = pd.read_parquet(REPO_ROOT / load_models()["cache"]["ledger"])
    led = led[~led["cache_hit"] & (led["timestamp"] >= run_start)]  # 今回の実行で課金された呼び出しのみ
    lines = ["# 合成パネル パイロット結果" + ("（スモーク）" if smoke else ""), "",
             f"SKU {skus.shape[0]}（IP {int((~skus.is_original).sum())} / オリジナル {int(skus.is_original.sum())}、2023–24 年発売）。"
             "条件 full / image_masked_text、ペルソナ 50、温度 0、seed 1、thinking なし。", "",
             "**精度の主張には使わない**（学習期間内でプロンプトを確かめるためのもの）。", "",
             "## 1. 回答の品質とコスト", "",
             "| モデル | 画像解像度 | 呼び出し | パース失敗 | 入力トークン/回 | 出力トークン/回 | USD/回 | USD 合計 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (mk, res), g in resp.groupby(["model_key", "media_resolution"]):
        lg = led[(led.model_key == mk) & (led.media_resolution == res)]
        lines.append(f"| {mk} | {res} | {len(g):,} | {g.parse_error.notna().mean():.1%} | "
                     f"{lg.input_tokens.mean():,.0f} | {lg.output_tokens.mean():,.0f} | ${lg.usd_cost.mean():.5f} | ${lg.usd_cost.sum():.2f} |")
    agg = aggregate(resp.drop(columns=["media_resolution"]).assign(
        model_key=resp["model_key"] + "/" + resp["media_resolution"]))
    agg = agg.merge(skus[["class_id", "color_id", "y", "is_original"]], on=["class_id", "color_id"])
    lines += ["", "## 2. ペルソナ間のばらつきと参考の順位相関", "",
              "intent_sd は SKU 内のペルソナ間標準偏差の平均（0 なら全員同じ答え）。ρ は主アウトカムとの Spearman（参考値）。", "",
              "| モデル | 条件 | intent 平均 | intent_sd | 全員同答の SKU | ρ(intent_mean) | ρ(prob_mean) | ρ IP 内 | ρ オリジナル内 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (mk, cond), g in agg.groupby(["model_key", "condition"]):
        ip, og = g[~g.is_original], g[g.is_original]
        lines.append(
            f"| {mk} | {cond} | {g.intent_mean.mean():.2f} | {g.intent_sd.mean():.2f} | {(g.intent_sd.fillna(0) == 0).mean():.0%} | "
            f"{spearman(g.y.values, g.intent_mean.values):.3f} | {spearman(g.y.values, g.prob_mean.values):.3f} | "
            f"{spearman(ip.y.values, ip.intent_mean.values):.3f} | {spearman(og.y.values, og.intent_mean.values):.3f} |")
    errs = resp[resp.parse_error.notna()].parse_error.value_counts().head(5)
    if len(errs):
        lines += ["", "## 3. パース失敗の内訳（上位）", ""] + [f"- {k}: {v}" for k, v in errs.items()]
    path = REPO_ROOT / "reports" / f"panel_pilot{'_smoke' if smoke else ''}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
