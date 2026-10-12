"""順位ベースの評価指標とクラスタ・ブートストラップ（CLAUDE.md §5.3）。

y は実績（主アウトカムの生の値）、p は予測スコア。いずれも 1 次元配列。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


def _is_constant(p: np.ndarray) -> bool:
    # np.std は同一値でも丸め誤差で 0 にならないことがある（B0 で発生）ので max − min で見る
    return bool(np.ptp(p) == 0)


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if _is_constant(p):
        return float("nan")
    return float(stats.spearmanr(y, p).statistic)


def _top(x: np.ndarray, n: int) -> np.ndarray:
    # 同点は index 順で安定に切る（予測が定数のときに結果が乱数に依存しないように）
    return np.argsort(-x, kind="stable")[:n]


def precision_at_k(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    """予測上位 k% のうち、実績でも上位 k% に入った割合。ランダムなら期待値 = frac。"""
    if _is_constant(p):
        return float("nan")
    n = max(1, math.ceil(frac * len(y)))
    return len(set(_top(p, n)) & set(_top(y, n))) / n


def ndcg_at_k(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    """ゲインは実績の販売点数そのもの（売れ筋を上位に置けたか）。"""
    if _is_constant(p):
        return float("nan")
    n = max(1, math.ceil(frac * len(y)))
    disc = 1 / np.log2(np.arange(2, n + 2))
    ideal = float(np.sort(y)[::-1][:n] @ disc)
    return float(y[_top(p, n)] @ disc) / ideal if ideal > 0 else float("nan")


def quantile_lift(y: np.ndarray, p: np.ndarray, q: int = 10) -> float:
    """予測上位 1/q の実績平均 ÷ 全体平均。"""
    if _is_constant(p):
        return float("nan")
    n = max(1, math.ceil(len(y) / q))
    return float(y[_top(p, n)].mean() / y.mean())


def all_metrics(y: np.ndarray, p: np.ndarray, fracs: list[float]) -> dict[str, float]:
    out = {"spearman": spearman(y, p), "lift_top10pct": quantile_lift(y, p, 10)}
    for f in fracs:
        pct = round(f * 100)
        out[f"precision_at_{pct}pct"] = precision_at_k(y, p, f)
        out[f"ndcg_at_{pct}pct"] = ndcg_at_k(y, p, f)
    return out


def cluster_bootstrap(
    df: pd.DataFrame,
    fn: Callable[[pd.DataFrame], dict[str, float]],
    cluster: str,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """クラスタ（graphic_id）単位で復元抽出し、fn の各指標の 95% 区間を返す。"""
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster).indices
    keys = np.array(list(groups))
    draws = []
    for _ in range(n_resamples):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        draws.append(fn(df.iloc[idx]))
    d = pd.DataFrame(draws)
    return pd.DataFrame({"lo": d.quantile(0.025), "hi": d.quantile(0.975)})
