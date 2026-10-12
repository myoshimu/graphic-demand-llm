import numpy as np
import pandas as pd

from src.eval.metrics import all_metrics, cluster_bootstrap, ndcg_at_k, precision_at_k, quantile_lift, spearman


def test_perfect_and_constant_predictions():
    y = np.arange(100, dtype=float)
    assert np.isclose(spearman(y, y), 1.0)
    assert precision_at_k(y, y, 0.1) == 1.0
    assert np.isclose(ndcg_at_k(y, y, 0.2), 1.0)
    assert quantile_lift(y, y) > 1.0
    const = np.zeros(100)
    assert all(np.isnan(v) for v in all_metrics(y, const, [0.1]).values())


def test_precision_counts_overlap_of_top_sets():
    y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    p = np.array([10, 1, 9, 2, 3, 4, 5, 6, 7, 8], dtype=float)  # 上位 2 = {0, 2}、実績上位 2 = {0, 1}
    assert precision_at_k(y, p, 0.2) == 0.5


def test_cluster_bootstrap_resamples_whole_clusters():
    df = pd.DataFrame({"g": ["a", "a", "b", "c"], "v": [1.0, 1.0, 2.0, 3.0]})
    seen = []
    ci = cluster_bootstrap(df, lambda x: seen.append(x["g"].value_counts().to_dict()) or {"m": x["v"].mean()}, "g", 50, 0)
    # "a" は常に 2 行の倍数で現れる
    assert all(c.get("a", 0) % 2 == 0 for c in seen)
    assert ci.loc["m", "lo"] <= ci.loc["m", "hi"]
