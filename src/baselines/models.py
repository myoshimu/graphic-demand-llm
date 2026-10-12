"""ベースライン B0–B4（CLAUDE.md §5.1）。

すべて「学習期間で fit → テスト期間を predict」。ハイパーパラメータは学習期間内の
GroupKFold（graphic_id）で選び、テスト期間は一切見ない。予測はターゲット（log1p）の尺度で返す。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = structlog.get_logger()


@dataclass
class Data:
    """学習・テストの特徴量。X_* はメタデータ（B1）、E_* は画像埋め込み（L2 正規化済み）。"""

    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    E_tr: np.ndarray
    E_te: np.ndarray
    y_tr: np.ndarray
    groups_tr: np.ndarray


def _cv_score(fit_predict, n: int, groups: np.ndarray, y: np.ndarray, folds: int) -> float:
    """GroupKFold の OOF 予測と実績の Spearman（主指標に合わせて順位で選ぶ）。"""
    oof = np.empty(n)
    for tr, va in GroupKFold(n_splits=folds).split(np.zeros(n), groups=groups):
        oof[va] = fit_predict(tr, va)
    return float(stats.spearmanr(y, oof).statistic)


# ---------------- 前処理 ----------------
def _meta_matrix(X_tr: pd.DataFrame, X_te: pd.DataFrame, num: list[str], cat: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """線形モデル用：数値は標準化、カテゴリは one-hot（学習期間に無い水準は全 0）。"""
    sc = StandardScaler().fit(X_tr[num])
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_tr[cat].astype(str))
    f = lambda X: np.hstack([sc.transform(X[num]), oh.transform(X[cat].astype(str))])  # noqa: E731
    return f(X_tr), f(X_te)


def _lgb_frame(X: pd.DataFrame, cat: list[str], cats: dict[str, pd.Index]) -> pd.DataFrame:
    out = X.copy()
    for c in cat:
        out[c] = pd.Categorical(out[c].astype(str), categories=cats[c])
    return out


# ---------------- B0 ----------------
def b0_constant(d: Data) -> np.ndarray:
    """カテゴリ内平均（主分析は T シャツ 1 カテゴリなので定数）。順位指標は定義できない。"""
    return np.full(len(d.X_te), d.y_tr.mean())


# ---------------- Ridge（B1 線形版 / B2）----------------
def ridge_fit_predict(A_tr: np.ndarray, A_te: np.ndarray, d: Data, cfg: dict, seed: int) -> tuple[np.ndarray, dict]:
    folds = cfg["cv_folds"]

    def run(alpha):
        def fp(tr, va):
            return Ridge(alpha=alpha).fit(A_tr[tr], d.y_tr[tr]).predict(A_tr[va])

        return _cv_score(fp, len(A_tr), d.groups_tr, d.y_tr, folds)

    scores = {a: run(a) for a in cfg["ridge"]["alphas"]}
    best = max(scores, key=scores.get)
    pred = Ridge(alpha=best).fit(A_tr, d.y_tr).predict(A_te)
    return pred, {"alpha": best, "cv_spearman": scores[best]}


# ---------------- LightGBM（B1 / B2）----------------
def lgb_fit_predict(F_tr: pd.DataFrame, F_te: pd.DataFrame, cat: list[str], d: Data, cfg: dict, seed: int) -> tuple[np.ndarray, dict]:
    lc = cfg["lightgbm"]
    grid = [dict(zip(lc["grid"], v, strict=True)) for v in itertools.product(*lc["grid"].values())]
    cat_cols = [c for c in cat if c in F_tr.columns]

    def model(params):
        return lgb.LGBMRegressor(**lc["fixed"], **params, random_state=seed, n_jobs=4)

    def run(params):
        def fp(tr, va):
            m = model(params).fit(F_tr.iloc[tr], d.y_tr[tr], categorical_feature=cat_cols)
            return m.predict(F_tr.iloc[va])

        return _cv_score(fp, len(F_tr), d.groups_tr, d.y_tr, cfg["cv_folds"])

    scores = [(run(p), p) for p in grid]
    best_score, best = max(scores, key=lambda t: t[0])
    pred = model(best).fit(F_tr, d.y_tr, categorical_feature=cat_cols).predict(F_te)
    return pred, {**best, "cv_spearman": best_score}


def b1_meta(d: Data, cfg: dict, seed: int, drop: tuple[str, ...] = ()) -> dict[str, tuple[np.ndarray, dict]]:
    num = [c for c in cfg["features"]["numeric"] if c not in drop]
    cat = [c for c in cfg["features"]["categorical"] if c not in drop]
    cats = {c: pd.Index(sorted(d.X_tr[c].astype(str).unique())) for c in cat}
    F_tr, F_te = _lgb_frame(d.X_tr[num + cat], cat, cats), _lgb_frame(d.X_te[num + cat], cat, cats)
    return {"lgbm": lgb_fit_predict(F_tr, F_te, cat, d, cfg, seed)}


def b2_image(d: Data, cfg: dict, seed: int) -> dict[str, tuple[np.ndarray, dict]]:
    num, cat = cfg["features"]["numeric"], cfg["features"]["categorical"]
    M_tr, M_te = _meta_matrix(d.X_tr, d.X_te, num, cat)
    out = {
        "emb_only_ridge": ridge_fit_predict(d.E_tr, d.E_te, d, cfg, seed),
        "ridge": ridge_fit_predict(np.hstack([d.E_tr, M_tr]), np.hstack([d.E_te, M_te]), d, cfg, seed),
    }
    # LightGBM 版：PCA は学習期間だけで fit
    k = cfg["lightgbm"]["embedding_pca_dims"]
    pca = PCA(n_components=k, random_state=seed).fit(d.E_tr)
    pcs = [f"pc{i:02d}" for i in range(k)]
    cats = {c: pd.Index(sorted(d.X_tr[c].astype(str).unique())) for c in cat}
    F_tr = pd.concat([_lgb_frame(d.X_tr[num + cat], cat, cats).reset_index(drop=True),
                      pd.DataFrame(pca.transform(d.E_tr), columns=pcs)], axis=1)
    F_te = pd.concat([_lgb_frame(d.X_te[num + cat], cat, cats).reset_index(drop=True),
                      pd.DataFrame(pca.transform(d.E_te), columns=pcs)], axis=1)
    out["lgbm"] = lgb_fit_predict(F_tr, F_te, cat, d, cfg, seed)
    return out


# ---------------- B3 ----------------
def b3_ip_scale(d: Data) -> np.ndarray:
    """IP 知名度プロキシ。Google Trends は未取得のため §5.1 の代替どおり ip_scale（オリジナル／不明も 1 水準）の学習期間平均。"""
    means = pd.Series(d.y_tr).groupby(d.X_tr["ip_scale"].astype(str).to_numpy()).mean()
    return d.X_te["ip_scale"].astype(str).map(means).fillna(d.y_tr.mean()).to_numpy()


# ---------------- B4 ----------------
def _knn_predict(E_ref: np.ndarray, y_ref: np.ndarray, E_q: np.ndarray, k: int) -> np.ndarray:
    sim = E_q @ E_ref.T
    idx = np.argpartition(-sim, kth=min(k, len(y_ref) - 1), axis=1)[:, :k]
    return y_ref[idx].mean(axis=1)


def b4_knn(d: Data, cfg: dict) -> tuple[np.ndarray, dict]:
    """埋め込み空間で近い過去デザイン k 点のターゲット平均。"""

    def run(k):
        def fp(tr, va):
            return _knn_predict(d.E_tr[tr], d.y_tr[tr], d.E_tr[va], k)

        return _cv_score(fp, len(d.E_tr), d.groups_tr, d.y_tr, cfg["cv_folds"])

    scores = {k: run(k) for k in cfg["knn"]["k_grid"]}
    best = max(scores, key=scores.get)
    return _knn_predict(d.E_tr, d.y_tr, d.E_te, best), {"k": best, "cv_spearman": scores[best]}
