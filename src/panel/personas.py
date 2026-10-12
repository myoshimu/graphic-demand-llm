"""ペルソナのサンプリングと文面生成（CLAUDE.md §5.2）。

config/personas.yaml の周辺分布に合わせ、軸ごとに最大剰余法で人数を割り当てる（quota）。
軸どうしは seed 付きシャッフルで独立に組み合わせる（同時分布は未取得）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONAS_PATH = REPO_ROOT / "config" / "personas.yaml"

AXES = ("age_band", "gender", "region", "purchase_frequency", "ip_fandom")

_AGE_JA = {"-19": "10代", "20-24": "20代前半", "25-29": "20代後半", "30-34": "30代前半",
           "35-39": "30代後半", "40-49": "40代", "50-": "50代以上"}
_GENDER_JA = {"female": "女性", "male": "男性"}
_REGION_JA = {"kanto": "関東", "kansai": "関西", "chubu": "中部", "kyushu": "九州・沖縄",
              "tohoku": "東北", "hokkaido": "北海道", "shikoku": "四国", "chugoku": "中国地方"}
_FREQ_JA = {
    "first_time": "retailer で T シャツを買ったことはほとんどありません",
    "occasional": "retailer の T シャツをたまに買います（数年に 2 回程度）",
    "regular": "retailer の T シャツを毎年のように買います",
    "heavy": "retailer の T シャツを年に何度も買う常連です",
}
_FANDOM_JA = {
    "none": "アニメ・キャラクターなどのコラボ商品にはあまり興味がなく、オリジナルデザインを好みます",
    "casual": "好きな作品やキャラクターのコラボ商品があれば買うこともあります",
    "fan": "好きな作品やキャラクターのコラボ商品をよく買います",
    "superfan": "好きな作品やキャラクターのコラボ商品を主に買う熱心なファンです",
}


class Persona(BaseModel):
    persona_id: int
    age_band: str
    gender: str
    region: str
    purchase_frequency: str
    ip_fandom: str

    def describe_ja(self) -> str:
        return (
            f"あなたは{_REGION_JA[self.region]}に住む{_AGE_JA[self.age_band]}の{_GENDER_JA[self.gender]}です。"
            f"{_FREQ_JA[self.purchase_frequency]}。{_FANDOM_JA[self.ip_fandom]}。"
        )


def load_config(path: Path = PERSONAS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def quota(weights: list[float], n: int) -> np.ndarray:
    """最大剰余法で n 人を重みに比例して割り当てる。"""
    w = np.asarray(weights, dtype=float)
    if w.min() < 0 or not np.isclose(w.sum(), 1.0, atol=1e-3):
        raise ValueError(f"重みが不正です: {weights}")
    raw = w / w.sum() * n
    counts = np.floor(raw).astype(int)
    rest = n - counts.sum()
    counts[np.argsort(-(raw - counts), kind="stable")[:rest]] += 1
    return counts


def sample_personas(n: int | None = None, cfg: dict[str, Any] | None = None) -> list[Persona]:
    cfg = cfg or load_config()
    n = n or int(cfg["n_personas"])
    rng = np.random.default_rng(int(cfg["sampling"]["seed"]) + n)
    cols: dict[str, list[str]] = {}
    for axis in AXES:
        spec = cfg["attributes"][axis]
        if spec.get("weights") is None:
            raise ValueError(f"config/personas.yaml の {axis}.weights が未設定です")
        vals = np.repeat(spec["levels"], quota(spec["weights"], n))
        cols[axis] = list(rng.permutation(vals))
    return [Persona(persona_id=i, **{a: cols[a][i] for a in AXES}) for i in range(n)]
