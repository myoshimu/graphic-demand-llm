#!/usr/bin/env python3
"""フェーズ0 のデータ品質チェックとアウトカム分布の可視化（CLAUDE.md §7-5 / §3.4）。

data/processed/{skus,sales_weekly,outcomes}.parquet を読んで
  - reports/figures/phase0_*.png
  - reports/phase0_data_quality.md
を出力する。BigQuery には触らない（build_tables.py の出力のみを入力とする）。

軸ラベル・タイトルは英語（論文用）。ただしカテゴリ名など BigQuery 由来の値は日本語のまま
出るため、japanize-matplotlib があれば読み込む。無ければ豆腐になるだけで処理は続行する。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

try:  # 日本語カテゴリ名を描画するため。未インストールでも致命的ではない
    import japanize_matplotlib  # noqa: F401,E402
except ImportError:
    print("WARN: japanize-matplotlib が無いため日本語ラベルは豆腐になります", file=sys.stderr)

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
FIGDIR = REPO / "reports" / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def fig_outcome_distributions(o: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), constrained_layout=True)

    for ax, w, c in [(axes[0, 0], 4, BLUE), (axes[0, 1], 8, ORANGE)]:
        u = o[f"units_{w}w"].clip(lower=0)
        ax.hist(np.log10(u + 1), bins=45, color=c, alpha=0.85)
        ax.set_xlabel(f"log10(units in first {w} weeks + 1)")
        ax.set_ylabel("SKUs")
        med = u.median()
        ax.axvline(np.log10(med + 1), color="k", ls="--", lw=1)
        ax.set_title(f"Units, {w}w window  (median {med:,.0f}, max {u.max():,.0f})", fontsize=9)

    ax = axes[1, 0]
    s = o["y_fullprice_share_8w"].dropna()
    ax.hist(s, bins=40, color=BLUE, alpha=0.85)
    ax.set_xlabel("full-price share of units, 8w")
    ax.set_ylabel("SKUs")
    ax.set_title(
        f"Full-price share  (median {s.median():.3f}, share==1.0: {(s >= 0.999).mean():.1%})",
        fontsize=9,
    )

    # ローレンツ曲線: アウトカムがどれだけ上位に偏っているか
    ax = axes[1, 1]
    for w, c in [(4, BLUE), (8, ORANGE)]:
        u = np.sort(o[f"units_{w}w"].clip(lower=0).to_numpy())
        cum = np.concatenate([[0], np.cumsum(u) / u.sum()])
        frac = np.linspace(0, 1, len(cum))
        ax.plot(frac, cum, color=c, lw=1.6, label=f"{w}w (Gini {gini(u):.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    ax.set_xlabel("cumulative share of SKUs (worst→best)")
    ax.set_ylabel("cumulative share of units")
    ax.set_title("Concentration of demand", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    u8 = o["units_8w"].clip(lower=0).to_numpy()
    top10 = np.sort(u8)[::-1][: max(1, len(u8) // 10)].sum() / u8.sum()
    return {"gini_8w": gini(u8), "top_decile_share_8w": float(top10)}


def fig_window_agreement(o: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    a, b = o["units_4w"].clip(lower=1), o["units_8w"].clip(lower=1)
    rho = stats.spearmanr(a, b).statistic
    ax = axes[0]
    ax.scatter(a, b, s=6, alpha=0.35, color=BLUE, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("units, first 4 weeks")
    ax.set_ylabel("units, first 8 weeks")
    ax.set_title(f"4w vs 8w  (Spearman rho = {rho:.3f})", fontsize=9)

    # n_stores_selling は配荷店舗数ではない、という証拠図
    ax = axes[1]
    ax.scatter(o["n_stores_selling_8w"], o["units_8w"].clip(lower=1), s=6, alpha=0.35,
               color=ORANGE, edgecolors="none")
    ax.set_yscale("log")
    ax.set_xlabel("# stores with >=1 unit sold, 8w")
    ax.set_ylabel("units, 8w")
    rho2 = stats.spearmanr(o["n_stores_selling_8w"], o["units_8w"]).statistic
    ax.set_title(
        f"'stores that sold' is an outcome, not a denominator (rho = {rho2:.2f})", fontsize=9
    )

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"spearman_4w_8w": float(rho), "spearman_units_nstores": float(rho2)}


def fig_sales_curve(w: pd.DataFrame, out: Path) -> dict:
    post = w[(w["week_index"] >= 0) & (w["week_index"] <= 25)]
    g = post.groupby("week_index")["units"]
    q = g.quantile([0.25, 0.5, 0.75]).unstack()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE, alpha=0.2, label="IQR")
    ax.plot(q.index, q[0.5], color=BLUE, lw=1.8, label="median")
    ax.set_xlabel("weeks since release")
    ax.set_ylabel("units sold that week")
    ax.set_title("Weekly sales decay", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    # 累積シェア: 8 週窓がライフタイムのどれだけを捕まえているか
    tot = w[w["week_index"] >= 0].groupby("class_id")["units"].sum()
    cum = (
        post.pivot_table(index="class_id", columns="week_index", values="units", aggfunc="sum")
        .reindex(columns=range(26))
        .fillna(0)
        .cumsum(axis=1)
    )
    share = cum.div(tot.reindex(cum.index).replace(0, np.nan), axis=0)
    ax = axes[1]
    ax.fill_between(share.columns, share.quantile(0.25), share.quantile(0.75),
                    color=ORANGE, alpha=0.2, label="IQR")
    ax.plot(share.columns, share.median(), color=ORANGE, lw=1.8, label="median")
    ax.axvline(7, color="k", ls="--", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("weeks since release")
    ax.set_ylabel("cumulative share of post-release units")
    ax.set_title("How much of lifetime the 8w window captures", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"median_share_captured_8w": float(share[7].median())}


def fig_channel(o: pd.DataFrame, out: Path) -> dict:
    """フェーズ0 の中心的な所見: units_8w の二峰性は需要ではなく流通幅である。

    左の峰は EC 専売 SKU（n_stores_selling==1）。合成パネルが予測するのは「魅力」なので、
    素の units_8w を正解にすると流通判断の当て物になってしまう。EC チャネルは全国一律に
    露出するため、EC 内の販売点数は配荷量に汚染されない需要指標として使える。
    """
    d = o.copy()
    d["ec_exclusive"] = d["n_stores_selling_8w"] <= 1

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    ax = axes[0]
    bins = np.linspace(0, 4, 45)
    for flag, lab, c in [(True, "EC-exclusive", ORANGE), (False, "store-distributed", BLUE)]:
        u = d.loc[d.ec_exclusive == flag, "units_8w"].clip(lower=0)
        ax.hist(np.log10(u + 1), bins=bins, color=c, alpha=0.6,
                label=f"{lab} (n={len(u)}, med {u.median():,.0f})")
    ax.set_xlabel("log10(units 8w + 1)")
    ax.set_ylabel("SKUs")
    ax.set_title("The two modes are two distribution regimes", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    ec = d.loc[d.units_ec_8w > 0, "units_ec_8w"]
    ax.hist(np.log10(ec), bins=40, color=BLUE, alpha=0.85)
    ax.set_xlabel("log10(EC-channel units, 8w)")
    ax.set_ylabel("SKUs")
    ax.set_title(f"EC-only outcome: unimodal (n={len(ec)}, med {ec.median():,.0f})", fontsize=9)

    ax = axes[2]
    sub = d[d.units_ec_8w > 0]
    rho = stats.spearmanr(sub.units_ec_8w, sub.units_8w).statistic
    ax.scatter(sub.units_ec_8w, sub.units_8w, s=6, alpha=0.35, color=GREY, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("EC-channel units, 8w")
    ax.set_ylabel("all-channel units, 8w")
    ax.set_title(f"EC vs all-channel ranking (rho = {rho:.2f})", fontsize=9)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_ec_exclusive": int(d.ec_exclusive.sum()),
        "median_units_ec_exclusive": float(d.loc[d.ec_exclusive, "units_8w"].median()),
        "median_units_store_distributed": float(d.loc[~d.ec_exclusive, "units_8w"].median()),
        "n_with_ec_sales": int((d.units_ec_8w > 0).sum()),
        "spearman_ec_vs_all": float(rho),
    }


def fig_segments(s: pd.DataFrame, o: pd.DataFrame, out: Path) -> None:
    d = s.merge(o, on="class_id")
    d["has_content_id"] = np.where(d["content_id"].fillna("") != "", "content_id present", "no content_id")
    d["release_month"] = pd.to_datetime(d["release_date"]).dt.month
    d["price_band"] = pd.cut(
        d["price_list"], [0, 3000, 3500, 4500, 1e9],
        labels=["<3000", "3000-3499", "3500-4499", ">=4500"],
    )

    specs = [
        ("has_content_id", "IP linkage"),
        ("ip_scale", "IP audience scale"),
        ("price_band", "Full price (JPY)"),
        ("design_category_name", "Design category"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), specs):
        sub = d[d[col].notna()].copy()
        counts = sub[col].value_counts()
        keep = counts[counts >= 10].index
        sub = sub[sub[col].isin(keep)]
        order = sub.groupby(col, observed=True)["units_8w"].median().sort_values().index
        data = [np.log10(sub.loc[sub[col] == k, "units_8w"].clip(lower=0) + 1) for k in order]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.45)
            patch.set_edgecolor(BLUE)
        for med in bp["medians"]:
            med.set_color("k")
        labels = [f"{k}  (n={int((sub[col] == k).sum())})" for k in order]
        ax.set_yticks(range(1, len(order) + 1), labels, fontsize=7)
        ax.set_xlabel("log10(units 8w + 1)")
        ax.set_title(title, fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(s: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), constrained_layout=True)

    m = pd.to_datetime(s["release_date"]).dt.to_period("M").value_counts().sort_index()
    axes[0].bar(range(len(m)), m.to_numpy(), color=BLUE, alpha=0.85)
    axes[0].set_xticks(range(len(m)), [str(p)[-2:] for p in m.index], fontsize=7)
    axes[0].set_xlabel("release month (2024)")
    axes[0].set_ylabel("SKUs (class_id)")
    axes[0].set_title(f"Release cadence  (n={len(s)})", fontsize=9)

    axes[1].hist(s["price_list"].dropna(), bins=40, color=ORANGE, alpha=0.85)
    axes[1].set_xlabel("full price (JPY)")
    axes[1].set_ylabel("SKUs")
    axes[1].set_title("Price", fontsize=9)

    ns = s["n_sizes"].value_counts().sort_index()
    axes[2].bar(ns.index.astype(int), ns.to_numpy(), color=GREY, alpha=0.85)
    axes[2].set_xlabel("# sizes sold")
    axes[2].set_ylabel("SKUs")
    axes[2].set_title("Size breadth", fontsize=9)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _q(x: pd.Series) -> str:
    return f"{x.median():,.0f} [{x.quantile(0.1):,.0f}–{x.quantile(0.9):,.0f}]"


def write_population_report(s: pd.DataFrame, o: pd.DataFrame, out: Path) -> None:
    """分析母集団（A31）に限ったデータ品質レポート。数値はすべてこのスクリプトが計算する。"""
    import yaml

    cfg = yaml.safe_load((REPO / "config" / "settings.yaml").read_text(encoding="utf-8"))
    y = cfg["outcomes"]["primary"]
    split = cfg["evaluation"]["split"]
    pop = pd.read_parquet(PROC / "population.parquet")
    gid = pd.read_parquet(PROC / "graphic_ids.parquet")

    cohort = s.merge(pop, on=["class_id", "color_id"], how="left")
    cohort["year"] = pd.to_datetime(cohort["release_date"]).dt.year
    d = cohort[cohort["in_population"].fillna(False).astype(bool)].copy()
    d = d.merge(o[["class_id", y]], on="class_id", how="left").merge(
        gid[["class_id", "color_id", "graphic_id", "review_uncertain"]], on=["class_id", "color_id"], how="left"
    )
    d["split"] = np.where(d["year"].isin(split["train"]), "train", np.where(d["year"].isin(split["test"]), "test", "-"))
    ip = ~d["is_original"].astype(bool)

    L = [
        "# フェーズ0 データ品質レポート（分析母集団）",
        "",
        "`make data-quality`（scripts/phase0_data_quality.py）が自動生成。手で編集しないこと。",
        "2024 年単年・全 SKU 版（2026-09-15）は `phase0_data_quality_2024.md` に保存。",
        "図（`reports/figures/phase0_*.png`）は母集団で絞る前の全コホートを描いている。",
        "",
        "## 1. 母集団の構成（A31）",
        "",
        "| 発売年 | コホート | not_listed | listed_late | listed_no_image | **母集団** |",
        "|---|---|---|---|---|---|",
    ]
    for yr, g in cohort.groupby("year"):
        r = g["exclusion_reason"].value_counts()
        L.append(
            f"| {yr} | {len(g)} | {r.get('not_listed', 0)} | {r.get('listed_late', 0)} | "
            f"{r.get('listed_no_image', 0)} | **{int(g['in_population'].fillna(False).sum())}** |"
        )
    L.append(f"| 計 | {len(cohort)} | | | | **{len(d)}** |")

    L += [
        "",
        "## 2. 年別の品質指標",
        "",
        f"主アウトカム `{y}`（中央値 [p10–p90]）。",
        "",
        "| 発売年 | split | n | 観測完了 | 正価欠損 | オリジナル | IP | IP 名欠損 | ip_scale 欠損 | content_id 欠損 | アウトカム | アウトカム 0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for yr, g in d.groupby("year"):
        gi = g[~g["is_original"].astype(bool)]
        L.append(
            f"| {yr} | {g['split'].iloc[0]} | {len(g)} | {int(g['fully_observed'].sum())} | "
            f"{int(g["price_list"].isna().sum())} | {int(g['is_original'].sum())} | {len(gi)} | "
            f"{int(gi['ip_name'].isna().sum())} | {int(gi['ip_scale'].isna().sum())} | "
            f"{int(g['content_id'].isna().sum())} | {_q(g[y])} | {int((g[y] == 0).sum())} |"
        )

    zero = d[d[y] == 0]
    L += ["", "### アウトカム 0 の SKU", ""]
    if len(zero):
        L += [f"- {r.class_id}（{r.year}、{r.content_name}）" for r in zero.itertuples()]
        L.append("- いずれも店舗では販売あり・EC 販売なし。現状は除外していない（0 のまま評価に入る）。")
    else:
        L.append("- なし")

    miss = d[ip & d["ip_scale"].isna()]
    L += [
        "",
        "## 3. IP タグの欠損（questions.md A32）",
        "",
        f"IP SKU {int(ip.sum())} 点中 {len(miss)} 点で `ip_scale` が欠損。年別は上表。"
        "テスト年に偏るため、B1/B3 と ip_scale 別の異質性分析は A32 の決着まで暫定扱い。",
        "",
        "| コンテンツ名 | 発売年 | SKU |",
        "|---|---|---|",
    ]
    for (name, yr), n in miss.fillna({"content_name": "（名前なし）"}).groupby(["content_name", "year"]).size().sort_values(ascending=False).items():
        L.append(f"| {name} | {yr} | {n} |")

    sizes = d.groupby("graphic_id").size()
    multi = sizes[sizes > 1]
    spans = d.groupby("graphic_id")["split"].nunique()
    span_ids = spans[spans > 1].index
    L += [
        "",
        "## 4. graphic_id（A30）",
        "",
        f"- 母集団 {len(d)} 点 → graphic_id {d['graphic_id'].nunique()} 個（欠損 {int(d['graphic_id'].isna().sum())}）",
        f"- 複数 SKU を持つグラフィック {len(multi)} 個・{int(multi.sum())} 点（最大 {int(sizes.max())} 点）",
        f"- 学習／テストをまたぐグラフィック {len(span_ids)} 個（テスト側 "
        f"{int(d[d['graphic_id'].isin(span_ids) & (d['split'] == 'test')].shape[0])} 点）。除外した感度分析を行う",
        f"- 判定保留（`review_uncertain`）{int(d['review_uncertain'].fillna(False).sum())} 点。暫定で「同一」扱い",
        "",
        "## 5. 学習／テスト",
        "",
        "| split | 発売年 | n | graphic_id | オリジナル | IP |",
        "|---|---|---|---|---|---|",
    ]
    for sp, yrs in [("train", split["train"]), ("test", split["test"])]:
        g = d[d["split"] == sp]
        L.append(
            f"| {sp} | {', '.join(map(str, yrs))} | {len(g)} | {g['graphic_id'].nunique()} | "
            f"{int(g['is_original'].sum())} | {int((~g['is_original'].astype(bool)).sum())} |"
        )
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(PROC / "skus.parquet")
    w = pd.read_parquet(PROC / "sales_weekly.parquet")
    o = pd.read_parquet(PROC / "outcomes.parquet")

    m: dict = {}
    m |= fig_outcome_distributions(o, FIGDIR / "phase0_outcome_distributions.png")
    m |= fig_window_agreement(o, FIGDIR / "phase0_window_agreement.png")
    m |= fig_sales_curve(w, FIGDIR / "phase0_sales_curve.png")
    m |= fig_channel(o, FIGDIR / "phase0_channel.png")
    fig_segments(s, o, FIGDIR / "phase0_segments.png")
    fig_coverage(s, FIGDIR / "phase0_coverage.png")

    u8 = o["units_8w"]
    lines = [
        f"n_skus={len(s)}",
        f"release={pd.to_datetime(s.release_date).min().date()}..{pd.to_datetime(s.release_date).max().date()}",
        f"fully_observed={int(s.fully_observed.sum())}/{len(s)}",
        f"units_8w median={u8.median():,.0f} mean={u8.mean():,.0f} max={u8.max():,.0f} min={u8.min():,.0f}",
        f"units_8w ratio p90/p10={u8.quantile(0.9) / max(u8.quantile(0.1), 1):,.1f}x",
        f"gini_8w={m['gini_8w']:.3f} top_decile_share={m['top_decile_share_8w']:.1%}",
        f"spearman(4w,8w)={m['spearman_4w_8w']:.3f}",
        f"spearman(units,n_stores_selling)={m['spearman_units_nstores']:.3f}",
        f"median share of post-release units captured by 8w window={m['median_share_captured_8w']:.1%}",
        f"skewness log1p(units_8w)={stats.skew(np.log1p(u8.clip(lower=0))):.3f}",
        f"no content_id={int((s.content_id.fillna('') == '').sum())}",
        f"has_preorder={int(s.has_preorder.sum())}",
        f"ec_exclusive={m['n_ec_exclusive']} (median units {m['median_units_ec_exclusive']:,.0f})"
        f" vs store_distributed median {m['median_units_store_distributed']:,.0f}",
        f"with_ec_sales={m['n_with_ec_sales']} spearman(ec,all)={m['spearman_ec_vs_all']:.3f}",
    ]
    print("\n".join(lines))
    (REPO / "reports" / "phase0_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_population_report(s, o, REPO / "reports" / "phase0_data_quality.md")


if __name__ == "__main__":
    sys.exit(main())
