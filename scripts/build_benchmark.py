"""公開ベンチマークを data/benchmark/ に生成（`make benchmark`）。内容は src/benchmark/build.py と config/benchmark.yaml。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_baselines import load_frame  # noqa: E402
from scripts.run_confirm import auto_graphic_ids  # noqa: E402
from src.benchmark.build import PUBLIC_PAIR_COLUMNS, PUBLIC_SKU_COLUMNS, build_pairs, build_skus, public  # noqa: E402
from src.ingest.bq import REPO_ROOT  # noqa: E402

PANEL = REPO_ROOT / "data" / "processed" / "panel"


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    bcfg = yaml.safe_load((REPO_ROOT / "config" / "baselines.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO_ROOT / "config" / "benchmark.yaml").read_text(encoding="utf-8"))
    sp = settings["evaluation"]["split"]
    split_of_year = {**{y: "train" for y in sp["train"]}, **{y: "test" for y in sp["test"]},
                     **{y: "confirm" for y in sp["confirm"]}}

    frame = load_frame(settings, bcfg, years=[*sp["train"], *sp["test"]])
    conf = load_frame(settings, bcfg, years=sp["confirm"], graphic_ids=auto_graphic_ids(settings, sp["confirm"]))
    frame = pd.concat([frame, conf[conf["fully_observed"]]], ignore_index=True)
    skus = build_skus(frame, cfg, split_of_year)

    pairs = build_pairs(pd.read_parquet(PANEL / "pairwise_all_pairs.parquet"), skus)
    out_pairs = public(pairs, PUBLIC_PAIR_COLUMNS)
    # 参考：本研究の LLM の 2 案比較スコア（sku_a を選ぶ確率、A/B 両順の平均）
    for run in cfg["reference_runs"]:
        if run == "pro_v0":
            rs = []
            for tag, year in (("", 2025), ("_confirm2026", 2026)):
                r = pd.read_parquet(PANEL / f"pairwise{tag}_responses.parquet")
                rs.append(r.assign(pair_id=f"{year}_" + r["pair"].astype(str)))
            r = pd.concat(rs)
        else:
            r = pd.read_parquet(PANEL / f"pairwise_rob_{run}.parquet")
        out_pairs[f"llm_{run}_p_a"] = out_pairs["pair_id"].map(r.groupby("pair_id").p_first.mean())
    pair_ids = {p: f"P{i:05d}" for i, p in enumerate(out_pairs["pair_id"])}
    out_pairs["pair_id"] = out_pairs["pair_id"].map(pair_ids)

    out = REPO_ROOT / settings["paths"]["benchmark"]
    out.mkdir(parents=True, exist_ok=True)
    files = {"skus.parquet": public(skus, PUBLIC_SKU_COLUMNS), "pairs.parquet": out_pairs}
    lines = []
    for name, df in files.items():
        df.to_parquet(out / name, index=False)
        lines.append(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")
        print(name, df.shape)
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    s = files["skus.parquet"]
    print(s.groupby("split").size().to_dict(), out_pairs.groupby("split").size().to_dict())
    print(s.sales_decile.value_counts().sort_index().to_dict(), s.price_band.value_counts().to_dict())


if __name__ == "__main__":
    main()
