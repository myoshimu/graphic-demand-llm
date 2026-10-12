"""論文用の図表 → paper/figures/*.pdf, paper/tables/*.tex（`make figures`）。

CLAUDE.md §2.1：IP 画像は一切載せない。画像を読み込むのは定性例（qualitative）だけで、オリジナル同士の組に限る（assert で確認）。
売上の実数も載せない。売上は年内の順位（パーセンタイル）か、2 案比較の正解率としてだけ使う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_panel_stage1 import crossfit  # noqa: E402
from scripts.eval_pairwise_robustness import boot, correct, load  # noqa: E402
from src.eval.metrics import cluster_bootstrap, spearman  # noqa: E402
from src.ingest.bq import REPO_ROOT  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
FIG = REPO_ROOT / "paper" / "figures"
TAB = REPO_ROOT / "paper" / "tables"
SETS = (("ip", "Within-IP pairs"), ("original", "Original-design pairs"))
METHODS = {
    "pro_v0": "Pro, expert prompt (primary)",
    "pro_v1": "Pro, paraphrased prompt",
    "flash_v0": "Flash, expert prompt",
    "flash_consumer": "Flash, 10 consumer personas",
    "claude46_v0": "Claude Sonnet 4.6, expert prompt",
    "B2": "B2: CLIP + metadata (LightGBM)",
    "B1": "B1: metadata only (LightGBM)",
}
COLORS = {"pro_v0": "C0", "pro_v1": "C9", "flash_v0": "C2", "flash_consumer": "C1", "claude46_v0": "C3", "B2": "0.4", "B1": "0.7"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig: plt.Figure, name: str) -> None:
    """PDF（論文用）と PNG（確認用）を出す。"""
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def settings() -> tuple[int, int]:
    s = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return int(s["evaluation"]["bootstrap"]["n_resamples"]), int(s["random_seed"])


def ci(t: pd.DataFrame, m: str, pct: bool = True) -> str:
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(t.at[m, 'acc'])} [{f(t.at[m, 'lo'])}, {f(t.at[m, 'hi'])}]"


def dci(t: pd.DataFrame, m: str) -> str:
    return f"{100 * t.at[m, 'd']:+.1f} [{100 * t.at[m, 'dlo']:+.1f}, {100 * t.at[m, 'dhi']:+.1f}]"


# ---- 段階 1：絶対評価のパネル ----
def stage1(n_bs: int, seed: int) -> None:
    base = pd.read_parquet(PROC / "baseline_predictions.parquet")
    sc = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet").query("condition == 'full'")
    df = base.merge(sc[["class_id", "color_id", "intent_mean"]], on=["class_id", "color_id"])
    df["y_pct"] = df.y_raw.rank(pct=True)
    df["oof_B2"] = crossfit(df, ["pred_B2_lgbm"])
    df["oof_B2_panel"] = crossfit(df, ["pred_B2_lgbm", "intent_mean"])

    # 序論の動機づけ図（1 段幅）：散布図だけ。グループ平均は大きな印と注記で示す（棒グラフは使わない）
    fig, ax = plt.subplots(figsize=(3.0, 1.75))
    grp = df.groupby("is_original").agg(intent=("intent_mean", "mean"), sales=("y_pct", "mean"))
    for orig, lab, c in ((False, "Licensed IP", "C3"), (True, "Original", "C0")):
        g = df[df.is_original == orig]
        ax.scatter(g.y_pct, g.intent_mean, s=4, alpha=0.45, c=c, label=f"{lab} (n={len(g)})", lw=0)
    for orig, c, tx, ha, name in ((False, "C3", 0.99, "right", "Licensed IP"), (True, "C0", 0.01, "left", "Original")):
        gx, gy = grp.at[orig, "sales"], grp.at[orig, "intent"]
        ax.scatter([gx], [gy], s=70, marker="X", c=c, edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(f"{name} mean:\nintent {gy:.2f}, sales pct. {100 * gx:.0f}", (gx, gy), xytext=(tx, 3.48), ha=ha, va="center",
                    fontsize=6, color="0.1", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    ax.set_xlabel("Actual 8-week online sales (percentile rank)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("Panel purchase intent (1-5)", fontsize=6.5, labelpad=1)
    ax.set_ylim(0.9, 3.7)
    ax.tick_params(labelsize=6, pad=1)
    ax.legend(frameon=False, fontsize=5.8, markerscale=2.5, loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_motivation")

    def stats(d: pd.DataFrame) -> dict[str, float]:
        y = d.y_raw.to_numpy(float)
        rb = spearman(y, d.oof_B2.to_numpy())
        return {"panel": spearman(y, d.intent_mean.to_numpy()), "B2": rb,
                "drho": spearman(y, d.oof_B2_panel.to_numpy()) - rb}

    # 商品名・IP 名を伏せた条件（画像と価格だけ）も同じ指標で並べる
    allsc = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet")
    rows = []
    for cond, title in (("full", "Image, product name, price, franchise (primary)"),
                        ("image_masked_text", "Image and price only (names hidden)")):
        dc = base.merge(allsc.query("condition == @cond")[["class_id", "color_id", "intent_mean"]], on=["class_id", "color_id"])
        dc["oof_B2"] = crossfit(dc, ["pred_B2_lgbm"])
        dc["oof_B2_panel"] = crossfit(dc, ["pred_B2_lgbm", "intent_mean"])
        rows.append(r"\multicolumn{5}{l}{\emph{" + title + r"}} \\")
        for name, sub in (("All", dc), ("Licensed IP", dc[~dc.is_original]), ("Original", dc[dc.is_original])):
            est, c = stats(sub), cluster_bootstrap(sub.reset_index(drop=True), stats, "graphic_id", n_bs, seed)
            f = lambda k: f"{est[k]:.3f} [{c.at[k, 'lo']:.3f}, {c.at[k, 'hi']:.3f}]"  # noqa: E731
            rows.append(r"\quad " + f"{name} & {len(sub)} & {f('panel')} & {f('B2')} & {f('drho')} " + r"\\")
    (TAB / "tab_stage1.tex").write_text("\n".join([
        r"\begin{tabular}{lrlll}", r"\toprule",
        r"Condition / subset & SKUs & Panel $\rho$ & B2 $\rho$ & $\Delta\rho$ (B2+panel) \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


def stage1_decomposition() -> None:
    """Study 1 の分散分解（式 9–11 の裏づけ、decisions_log 2026-10-11（9））→ paper/tables/stage1_decomp.tex（本文の数値の出典）。"""
    from scipy.stats import spearmanr

    base = pd.read_parquet(PROC / "baseline_predictions.parquet")
    sc = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet").query("condition == 'full'")
    sk = pd.read_parquet(PROC / "skus.parquet").drop_duplicates(["class_id", "color_id"])[["class_id", "color_id", "content_name", "release_date"]]
    df = base.merge(sc[["class_id", "color_id", "intent_mean"]], on=["class_id", "color_id"]).merge(sk, on=["class_id", "color_id"])
    df["ly"] = np.log1p(df.y_raw.astype(float))
    # グループは Study 2 と同じ：ライセンスは IP、オリジナルは発売月
    df["g"] = np.where(df.is_original, "orig_" + pd.to_datetime(df.release_date).dt.strftime("%Y-%m"), "ip_" + df.content_name.astype(str))

    def between_share(col: str) -> float:
        gm = df.groupby("g")[col].transform("mean")
        return float(((gm - df[col].mean()) ** 2).sum() / ((df[col] - df[col].mean()) ** 2).sum())

    grp = df.groupby("g").agg(ly=("ly", "mean"), s=("intent_mean", "mean"), n=("ly", "size"))
    grp = grp[grp.n >= 3]
    w_ly = df.ly - df.groupby("g").ly.transform("mean")
    w_s = df.intent_mean - df.groupby("g").intent_mean.transform("mean")
    def share_in(mask: pd.Series) -> float:
        sub = df[mask]
        gm = sub.groupby("g").ly.transform("mean")
        return float(((gm - sub.ly.mean()) ** 2).sum() / ((sub.ly - sub.ly.mean()) ** 2).sum())

    vals = {"DecBetweenSales": f"{100 * between_share('ly'):.0f}",
            "DecShareFranchise": f"{100 * share_in(~df.is_original):.0f}", "DecShareMonth": f"{100 * share_in(df.is_original):.0f}", "DecBetweenScore": f"{100 * between_share('intent_mean'):.0f}",
            "DecGroups": f"{len(grp)}", "DecGroupRho": f"{spearmanr(grp.ly, grp.s)[0]:.2f}", "DecWithinRho": f"{spearmanr(w_ly, w_s)[0]:.2f}"}
    (TAB / "stage1_decomp.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()) + "\n", encoding="utf-8")
    print("stage1_decomp", vals)


def stage1_other_outputs(n_bs: int, seed: int) -> None:
    """Study 1 の未使用の回答（購入確率、支払上限額÷価格）も同じ指標で評価 → tables/stage1_other.tex（本文の数値の出典）。
    decisions_log 2026-10-12（8）。"""
    base = pd.read_parquet(PROC / "baseline_predictions.parquet")
    sc = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet").query("condition == 'full'")
    resp = pd.read_parquet(PROC / "panel" / "stage1_responses.parquet").query("condition == 'full'")
    price = pd.read_parquet(PROC / "skus.parquet").drop_duplicates(["class_id", "color_id"])[["class_id", "color_id", "price_list"]]
    resp = resp.merge(price, on=["class_id", "color_id"])
    resp["wtp_ratio"] = resp.would_pay_jpy / resp.price_list  # 価格の影響を除くため正価で割る
    wtp = resp.groupby(["class_id", "color_id"]).wtp_ratio.median().rename("wtp_ratio").reset_index()
    df = base.merge(sc[["class_id", "color_id", "prob_mean"]], on=["class_id", "color_id"]).merge(wtp, on=["class_id", "color_id"])
    df["oof_B2"] = crossfit(df, ["pred_B2_lgbm"])
    out = {}
    for m, tag in (("prob_mean", "Prob"), ("wtp_ratio", "Wtp")):
        df["oof_x"] = crossfit(df, ["pred_B2_lgbm", m])

        def stats(d: pd.DataFrame, m: str = m) -> dict[str, float]:
            y = d.y_raw.to_numpy(float)
            return {"rho": spearman(y, d[m].to_numpy()), "drho": spearman(y, d.oof_x.to_numpy()) - spearman(y, d.oof_B2.to_numpy())}

        for name, sub in (("All", df), ("IP", df[~df.is_original])):
            est = stats(sub)
            c = cluster_bootstrap(sub.reset_index(drop=True), stats, "graphic_id", n_bs, seed)
            for k in ("rho", "drho"):
                out[f"{tag}{name}{k.capitalize()}"] = f"{est[k]:+.3f} [{c.at[k, 'lo']:+.3f}, {c.at[k, 'hi']:+.3f}]"
    (TAB / "stage1_other.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()) + "\n", encoding="utf-8")
    print("stage1_other", out)


# ---- 2 案比較 ----
def boot_shared(g: pd.DataFrame, cols: list[str], n_bs: int, seed: int, save_as: str) -> pd.DataFrame:
    """全手法に同じクラスタ再抽出を当てるブートストラップ（表 2 の合算の行と手法の違いの行、decisions_log 2026-10-12（18））。

    再抽出したクラスタは data/processed/panel/bootstrap_draws_{save_as}.npz に保存する。B2 との差も同じ再抽出で計算する。
    """
    C = g[[f"c_{c}" for c in cols]].to_numpy()
    cl = g["cluster"].to_numpy()
    keys = np.unique(cl)
    idx = {k: np.where(cl == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(keys, len(keys)) for _ in range(n_bs)])
    np.savez_compressed(PROC / "panel" / f"bootstrap_draws_{save_as}.npz", keys=keys, draws=draws)
    bs = np.array([np.nanmean(C[np.concatenate([idx[k] for k in row])], 0) for row in draws])
    est = np.nanmean(C, 0)
    j = cols.index("B2")
    return pd.DataFrame({
        "n": (~np.isnan(C)).sum(0), "acc": est, "lo": np.nanpercentile(bs, 2.5, 0), "hi": np.nanpercentile(bs, 97.5, 0),
        "d": est - est[j], "dlo": np.nanpercentile(bs - bs[:, [j]], 2.5, 0), "dhi": np.nanpercentile(bs - bs[:, [j]], 97.5, 0),
    }, index=cols)


def prereg_report_rows(report: Path) -> dict[str, dict[str, str]]:
    """事前登録した評価コード（scripts/eval_pairwise.py）の出力 reports/*.md から、2 案比較と B2 の正解率・差の区間を読む。
    表 2 の 2025・2026 年の行は、合格判定と同じ数値を出すために計算し直さずこれを使う（decisions_log 2026-10-12（14））。"""
    import re

    txt = report.read_text(encoding="utf-8")
    out = {}
    for key, head in (("ip", "## IP 内"), ("original", "## オリジナル")):
        sec = txt[txt.index(head):]
        sec = sec[: sec.find("\n## ", 5) if sec.find("\n## ", 5) > 0 else len(sec)]
        n, k = re.search(r"（(\d+) 組、(\d+) グループ", sec).groups()
        pro = re.search(r"\| 2 案比較（Pro） \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \|", sec).groups()
        b2 = re.search(r"\| B2_lgbm \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \|", sec).groups()
        pct = lambda x: f"{100 * float(x):.1f}"  # noqa: E731
        sg = lambda x: f"{100 * float(x):+.1f}"  # noqa: E731
        out[key] = {"n": n, "k": k, "pro": f"{pct(pro[0])} [{pct(pro[1])}, {pct(pro[2])}]",
                    "b2": f"{pct(b2[0])} [{pct(b2[1])}, {pct(b2[2])}]", "d": f"{sg(pro[3])} [{sg(pro[4])}, {sg(pro[5])}]"}
    return out


def pairwise(d: pd.DataFrame, n_bs: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    methods = [m for m in METHODS if m in d]
    for m in methods:
        d[f"c_{m}"] = correct(d[m], d["truth"])

    # 表：年ごと（2025 = 探索、2026 = 事前登録した確認）と合算、主結果と B2
    prereg = {"2025": prereg_report_rows(REPO_ROOT / "reports" / "pairwise.md"),
              "2026": prereg_report_rows(REPO_ROOT / "reports" / "pairwise_confirm2026.md")}
    # 表 2（主結果と頑健性を統合、decisions_log 2026-10-12（15））。計算の順序は統合前と同じにして、乱数の並び＝区間の値を変えない
    main_rows: dict[str, list[str]] = {"ip": [], "original": []}
    shared: dict[str, pd.DataFrame] = {}
    for st, title in SETS:
        for lab, sub in (("2025", d[(d["set"] == st) & (d.year == 2025)]),
                         ("2026", d[(d["set"] == st) & (d.year == 2026)]),
                         ("pooled", d[d["set"] == st])):
            t = boot(sub, ["pro_v0", "B2"], n_bs, rng)  # 合算の行に使う。年の行でも呼んで乱数の並びを変えない
            if lab != "pooled":
                r = prereg[lab][st]  # 年の行は事前登録した評価コードの出力（合格判定と同じ数値）
                assert int(r["n"]) == len(sub), (lab, st, r["n"], len(sub))
                main_rows[st].append(f"\\quad Gemini 3.1 Pro, {lab} & {r['n']} & {r['k']} & {r['pro']} & {r['b2']} & {r['d']} \\\\")
                continue
            pooled_cols = [m for m in ("pro_v0", "pro_v1", "flash_v0", "flash_consumer", "claude46_v0", "B1", "B2") if m in sub and sub[m].notna().any()]
            shared[st] = boot_shared(sub, pooled_cols, n_bs, seed + 10 + (st == "original"), st)
            t = shared[st]
            main_rows[st].append(f"\\quad Gemini 3.1 Pro, pooled & {len(sub)} & {sub.cluster.nunique()} & {ci(t, 'pro_v0')} & {ci(t, 'B2')} & {dci(t, 'pro_v0')} \\\\")
        if st == "ip":
            # 価格も同じ組（文面が同じで画像だけが違う）。既存の行の区間を変えないよう別の乱数
            sp = d[(d["set"] == "ip") & (d.first_price_list == d.second_price_list)]
            t = boot(sp, ["pro_v0", "B2"], n_bs, np.random.default_rng(seed + 1))
            main_rows[st].append(f"\\qquad same price only & {len(sp)} & {sp.cluster.nunique()} & {ci(t, 'pro_v0')} & {ci(t, 'B2')} & {dci(t, 'pro_v0')} \\\\")
            # 同じモデル・同じ商品で、Study 1 の絶対評価（購買意向の平均）の差で並べた場合（2025 年の組のみ）
            sc1 = pd.read_parquet(PROC / "panel" / "stage1_scores.parquet").query("condition == 'full'").set_index(["class_id", "color_id"]).intent_mean
            g25 = d[(d["set"] == "ip") & (d.year == 2025)].copy()
            g25["abs"] = [sc1.get((a, b), np.nan) - sc1.get((c, e), np.nan) for a, b, c, e in
                          zip(g25.first_class_id, g25.first_color_id, g25.second_class_id, g25.second_color_id)]
            g25["c_abs"] = correct(g25["abs"], g25["truth"])
            t = boot(g25, ["abs", "B2"], n_bs, np.random.default_rng(seed + 2))
            main_rows[st].append(f"\\qquad absolute ratings (Study 1), 2025 & {len(g25)} & {g25.cluster.nunique()} & {ci(t, 'abs')} & {ci(t, 'B2')} & {dci(t, 'abs')} \\\\")
            # 2 案比較 − 絶対評価（同じ組・同じ再標本）。本文の数値の出典
            fd = boot(g25.assign(c_B2=g25.c_abs, B2=g25["abs"]), ["pro_v0", "B2"], n_bs, np.random.default_rng(seed + 3))
            (TAB / "format_diff.tex").write_text(
                f"\\newcommand{{\\FmtAbs}}{{{100 * fd.at['B2', 'acc']:.1f}}}\\newcommand{{\\FmtPair}}{{{100 * fd.at['pro_v0', 'acc']:.1f}}}"
                f"\\newcommand{{\\FmtDiff}}{{{dci(fd, 'pro_v0')}}}\n", encoding="utf-8")

    # 頑健性（合算）。費用と速度は表から外して本文に書く（数値の出典は tables/cost_latency.tex）
    led = pd.read_parquet(REPO_ROOT / "data" / "cache" / "llm_ledger.parquet")
    led = led[~led.cache_hit & led.image_sha256.fillna("").str.contains(r"\+")]
    per_call = led.groupby("model_key").agg(usd=("usd_cost", "mean"), lat=("latency_seconds", "median"))
    model_of = {"pro_v0": "api_a", "flash_v0": "api_b", "flash_consumer": "api_b", "claude46_v0": "api_c"}
    calls_per_pair = {"flash_consumer": 10}  # 消費者 10 人に 1 回ずつ。他は順序を入れ替えて 2 回
    (TAB / "cost_latency.tex").write_text("".join(
        f"\\newcommand{{\\Cost{tag}}}{{\\${calls_per_pair.get(m, 2) * per_call.at[model_of[m], 'usd']:.3f}}}"
        f"\\newcommand{{\\Lat{tag}}}{{{per_call.at[model_of[m], 'lat']:.1f}}}"
        for m, tag in (("pro_v0", "Pro"), ("flash_v0", "Flash"), ("flash_consumer", "Consumer"), ("claude46_v0", "Claude"))) + "\n",
        encoding="utf-8")
    variant_label = {"pro_v1": "Pro, paraphrased prompt", "flash_v0": "Flash, expert prompt", "flash_consumer": "Flash, 10 consumer personas",
                     "claude46_v0": "Claude Sonnet 4.6", "B1": "B1: metadata only"}
    var_rows: dict[str, list[str]] = {"ip": [], "original": []}
    res = {}
    for st, title in SETS:
        g = d[d["set"] == st]
        for m in methods:
            if g[m].isna().all():
                continue
            sub = g[g[m].notna()]
            t = boot(sub, [m, "B2"] if m != "B2" else ["B2"], n_bs, rng)
            res[(st, m)] = t.loc[m]
            if m in variant_label:
                assert len(sub) == len(g), m  # 手法の違いの行は合算の行と同じ組（同じ再抽出を当てるため）
                ts = shared[st]
                var_rows[st].append(f"\\quad {variant_label[m]} & {len(sub)} & {sub.cluster.nunique()} & {ci(ts, m)} & -- & {dci(ts, m)} \\\\")
    body = []
    for st, title in SETS:
        body += [rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\", *main_rows[st],
                 r"\addlinespace\multicolumn{6}{l}{\textit{pooled variants}} \\", *var_rows[st], r"\midrule"]
    (TAB / "tab_pairwise_main.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrrlll@{}}", r"\toprule",
        r"Condition & $n$ & Cl. & LLM (\%) & B2 (\%) & Diff.\ (pt) \\", r"\midrule",
        *body[:-1], r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharex=True, sharey=True)
    for ax, (st, title) in zip(axes, SETS):
        for i, m in enumerate(methods):
            if (st, m) not in res:
                continue
            r = res[(st, m)]
            ax.errorbar(100 * r.acc, i, xerr=[[100 * (r.acc - r.lo)], [100 * (r.hi - r.acc)]], fmt="o",
                        color=COLORS[m], ms=4, capsize=2)
        ax.axvline(50, color="0.8", lw=0.8, ls="--")
        ax.set(title=title, xlabel="Pairwise accuracy (%)")
    axes[0].set(yticks=range(len(methods)), yticklabels=[METHODS[m] for m in methods])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig_pairwise_accuracy")

    # 図：選択的予測（確信度の高い順に判定する組を増やしたときの正解率）
    cov = np.linspace(0.1, 1.0, 19)
    # 1 段幅の図（論文では確信度の当てはまり fig_calibration と別の図）
    # 採択論文に合わせて小さく（1 段幅 3.3×1.85 インチ）。線は主要 4 本、言い換え版と消費者役は表 3 にある
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.85), sharey=True)
    for ax, (st, title) in zip(axes, SETS):
        g = d[d["set"] == st]
        for m in methods:
            if m in ("B1", "pro_v1", "flash_consumer") or g[m].isna().all():
                continue
            c = g.loc[g[m].abs().sort_values(ascending=False, kind="stable").index, f"c_{m}"].to_numpy()
            acc = [c[: max(1, int(round(k * len(c))))].mean() for k in cov]
            short = {"pro_v0": "Pro (primary)", "pro_v1": "Pro, paraphrased", "flash_v0": "Flash",
                     "flash_consumer": "Flash, consumer personas", "claude46_v0": "Claude Sonnet 4.6", "B2": "B2"}
            ax.plot(100 * cov, 100 * np.array(acc), color=COLORS[m], lw=1.6 if m == "pro_v0" else 1,
                    label=short.get(m, METHODS[m]))
        ax.axhline(50, color="0.8", lw=0.8, ls="--")
        ax.set_title(title.replace(" pairs", ""), fontsize=7, pad=2)
        ax.set_xlabel("Coverage (%)", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6, pad=1)
    axes[0].set_ylabel("Accuracy (%)", fontsize=6.5, labelpad=1)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=5.8, ncol=4, loc="upper center",
               handlelength=1.4, columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.3, w_pad=0.6)
    save(fig, "fig_selective")


def mask_logo(img: "Image.Image") -> "Image.Image":
    """投稿版（ダブルブラインド）用：商品画像の右上にある小売企業のロゴを背景色で塗りつぶす（decisions_log 2026-10-10（5））。"""
    from PIL import ImageDraw

    img = img.convert("RGB")
    w, h = img.size
    bg = img.getpixel((int(w * 0.75), int(h * 0.02)))
    ImageDraw.Draw(img).rectangle([int(w * 0.80), 0, w, int(h * 0.15)], fill=bg)
    return img


def select_qualitative(d: pd.DataFrame, seed: int = 20261008) -> pd.DataFrame:
    """定性例の組（オリジナル同士、Pro の確信度上位 25% から正解 2・不正解 1）。"""
    o = d[d["set"] == "original"].copy()
    o["c"] = correct(o["pro_v0"], o["truth"])
    top = o[o["pro_v0"].abs().rank(pct=True, method="first") > 0.75]
    sel = pd.concat([top[top.c == 1].sample(2, random_state=seed), top[top.c == 0].sample(1, random_state=seed)])
    assert sel.first_is_original.all() and sel.second_is_original.all(), "IP 画像を図に載せない（§2.1）"
    return sel


def qualitative(d: pd.DataFrame, seed: int = 20261008) -> None:
    """定性例：オリジナル同士の組で、Pro の確信度が上位 25% の組から正解 2・不正解 1 を seed 付きで無作為に選ぶ。

    選び方は結果を見る前に決めた（decisions_log 2026-10-08）。理由文の英訳は本文のキャプションに書く。
    """
    from PIL import Image

    sel = select_qualitative(d, seed)
    man = pd.read_parquet(REPO_ROOT / "data" / "cache" / "image_manifest.parquet").set_index(["class_id", "color_id"])
    fig, axes = plt.subplots(1, 2 * len(sel), figsize=(7.0, 1.95))
    for i, (pid, r) in enumerate(sel.iterrows()):
        llm_first = r.pro_v0 > 0
        for j, side in enumerate(("first", "second")):
            ax = axes[2 * i + j]
            ax.imshow(mask_logo(Image.open(REPO_ROOT / man.loc[(r[f"{side}_class_id"], r[f"{side}_color_id"]), "image_path"])))
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set_visible(False)
            sold = (r.truth == 1) == (side == "first")
            picked = llm_first == (side == "first")
            tags = [t for t, on in (("sold more", sold), ("LLM pick", picked)) if on]
            ax.set_title(f"({chr(97 + i)}{j + 1})\n" + (", ".join(tags) or " "), fontsize=7,
                         color="C2" if sold and picked else ("C3" if picked else "0.2"))
    fig.tight_layout()
    save(fig, "fig_qualitative")


# ---- 探索的分析（結果を見た後の分析。decisions_log 2026-10-11（3））----
def exploratory(d: pd.DataFrame, n_bs: int, seed: int) -> None:
    """(1) 価格も同じ組（画像だけが違う）での正解率、(2) 売上差の四分位別の正解率、(3) 同じ IP の候補から 1 点選ぶ評価、(4) IP 規模別（付録）。"""
    rng = np.random.default_rng(seed)
    for m in ("pro_v0", "B2"):
        d[f"c_{m}"] = correct(d[m], d["truth"])
    d["same_price"] = d.first_price_list == d.second_price_list
    d["gap"] = (np.log1p(d.first_y_raw.astype(float)) - np.log1p(d.second_y_raw.astype(float))).abs()

    # (1) 価格が同じ組／違う組
    rows = []
    for st, title in SETS:
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{title}}}}} \\")
        for lab, sub in (("Same price", d[(d["set"] == st) & d.same_price]),
                         ("Different prices", d[(d["set"] == st) & ~d.same_price])):
            t = boot(sub, ["pro_v0", "B2"], n_bs, rng)
            rows.append(f"\\quad {lab} & {len(sub)} & {ci(t, 'pro_v0')} & {ci(t, 'B2')} \\\\")
    (TAB / "tab_same_price.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrll@{}}", r"\toprule",
        r"Pairs & $n$ & Gemini Pro (\%) & B2 (\%) \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    # (2) 売上差の四分位別
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.6), sharey=True)
    for ax, (st, title) in zip(axes, SETS):
        g = d[d["set"] == st].copy()
        g["q"] = pd.qcut(g.gap, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for off, m, col, lab in ((-0.12, "pro_v0", COLORS["pro_v0"], "Gemini Pro"), (0.12, "B2", COLORS["B2"], "B2")):
            est = [boot(g[g.q == q], ["pro_v0", "B2"], n_bs, rng).loc[m] for q in ["Q1", "Q2", "Q3", "Q4"]]
            x = np.arange(4) + off
            ax.errorbar(x, [100 * e.acc for e in est], yerr=[[100 * (e.acc - e.lo) for e in est], [100 * (e.hi - e.acc) for e in est]],
                        fmt="o-", color=col, ms=3, lw=1, capsize=1.5, label=lab)
        ax.axhline(50, color="0.8", lw=0.8, ls="--")
        ax.set(xticks=range(4), xticklabels=["Q1", "Q2", "Q3", "Q4"])
        ax.set_title(title.replace(" pairs", ""), fontsize=7, pad=2)
        ax.set_xlabel("Sales-gap quartile", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6, pad=1)
    axes[0].set_ylabel("Accuracy (%)", fontsize=6.5, labelpad=1)
    axes[0].legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "fig_gap")

    # (3) 同じ IP（×年）の候補から 1 点選ぶ：勝率（式 5 の平均）が最大の SKU と、B2 の予測が最大の SKU
    recs = []
    for (grp, yr), g in d[d["set"] == "ip"].groupby(["group", "year"]):
        a = g.first_class_id + "." + g.first_color_id
        b = g.second_class_id + "." + g.second_color_id
        sales = pd.concat([pd.Series(g.first_y_raw.values, a.values), pd.Series(g.second_y_raw.values, b.values)])
        sales = sales[~sales.index.duplicated()].astype(float)
        if len(sales) < 3:
            continue
        p = g.pro_v0.to_numpy() + 0.5
        win = pd.concat([pd.Series(p, a.values), pd.Series(1 - p, b.values)]).groupby(level=0).mean()
        b2 = pd.concat([pd.Series(g.first_pred_B2_lgbm.values, a.values), pd.Series(g.second_pred_B2_lgbm.values, b.values)])
        b2 = b2[~b2.index.duplicated()]
        pct = sales.rank(pct=True)
        best = sales.max()
        recs.append({"cluster": f"{grp}_{yr}", "n": len(sales),
                     "llm_top": float(sales[win.idxmax()] == best), "b2_top": float(sales[b2.idxmax()] == best),
                     "rand_top": float((sales == best).sum() / len(sales)),
                     "llm_pct": pct[win.idxmax()], "b2_pct": pct[b2.idxmax()], "rand_pct": pct.mean()})
    r = pd.DataFrame(recs)
    cols = ["llm_top", "b2_top", "rand_top", "llm_pct", "b2_pct", "rand_pct"]
    bs = np.array([r.iloc[rng.integers(0, len(r), len(r))][cols].mean().to_numpy() for _ in range(n_bs)])
    est = r[cols].mean().to_numpy()
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    f = lambda k: f"{100 * est[k]:.1f} [{100 * lo[k]:.1f}, {100 * hi[k]:.1f}]"  # noqa: E731
    dlt = lambda k, j: f"{100 * (est[k] - est[j]):+.1f} [{100 * np.percentile(bs[:, k] - bs[:, j], 2.5):+.1f}, {100 * np.percentile(bs[:, k] - bs[:, j], 97.5):+.1f}]"  # noqa: E731
    (TAB / "tab_top1.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lll@{}}", r"\toprule",
        r"Pick & Best seller (\%) & Percentile of pick \\", r"\midrule",
        f"Gemini Pro & {f(0)} & {f(3)} \\\\",
        f"B2 & {f(1)} & {f(4)} \\\\",
        f"Random & {f(2)} & {f(5)} \\\\",
        r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")
    (TAB / "top1_meta.tex").write_text(f"\\newcommand{{\\TopGroups}}{{{len(r)}}}\\newcommand{{\\TopCands}}{{{r.n.mean():.1f}}}\n", encoding="utf-8")

    # (4) IP 規模別（付録）
    sc = pd.read_parquet(PROC / "skus.parquet").drop_duplicates(["class_id", "color_id"])[["class_id", "color_id", "ip_scale"]]
    ip = d[d["set"] == "ip"].merge(sc.add_prefix("first_"), on=["first_class_id", "first_color_id"], how="left")
    names = {"ニッチ": "Niche", "中堅": "Mid-tier", "メジャー": "Major", "グローバルメガ": "Global mega"}
    rows = []
    for k, en in names.items():
        sub = ip[ip.first_ip_scale == k]
        t = boot(sub, ["pro_v0", "B2"], n_bs, rng)
        rows.append(f"{en} & {len(sub)} ({sub.cluster.nunique()}) & {ci(t, 'pro_v0')} & {ci(t, 'B2')} \\\\")
    rows.append(f"Not tagged & {int(ip.first_ip_scale.isna().sum())} ({ip[ip.first_ip_scale.isna()].cluster.nunique()}) & "
                f"{100 * ip[ip.first_ip_scale.isna()].c_pro_v0.mean():.1f} & {100 * ip[ip.first_ip_scale.isna()].c_B2.mean():.1f} \\\\")
    (TAB / "tab_scale.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrll@{}}", r"\toprule",
        r"Scale & Pairs (fr.) & Gemini Pro (\%) & B2 (\%) \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


def extras(d: pd.DataFrame, n_bs: int, seed: int) -> None:
    """(a) 確信度の当てはまり（信頼性図）、(c) 公開ベンチマークの概要（(b) 費用と速度は表 3 に統合、pairwise() 内）。decisions_log 2026-10-11（4）。"""
    rng = np.random.default_rng(seed)
    d["c_pro_v0"] = correct(d["pro_v0"], d["truth"])
    d["c_B2"] = correct(d["B2"], d["truth"])
    d["conf"] = d["pro_v0"].abs() + 0.5
    d["bin"] = pd.cut(d["conf"], [0.5, 0.55, 0.7, 0.775, 0.85, 1.0], include_lowest=True)
    fig, ax = plt.subplots(figsize=(3.3, 1.65))
    for st, title, col, mk in (("ip", "Within-IP", "C0", "o"), ("original", "Original-design", "C1", "s")):
        g = d[d["set"] == st]
        xs, ys, lo, hi = [], [], [], []
        for _, gb in g.groupby("bin", observed=True):
            if len(gb) < 20:
                continue
            t = boot(gb, ["pro_v0", "B2"], n_bs, rng).loc["pro_v0"]
            xs.append(gb.conf.mean()); ys.append(t.acc); lo.append(t.acc - t.lo); hi.append(t.hi - t.acc)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk + "-", color=col, ms=3, lw=1, capsize=1.5, label=title)
    ax.plot([0.5, 0.9], [0.5, 0.9], color="0.6", ls="--", lw=0.8, label="Perfect calibration")
    ax.set(xlim=(0.48, 0.92), ylim=(0.4, 0.92))
    ax.set_xlabel("Stated confidence", fontsize=6.5, labelpad=1)
    ax.set_ylabel("Accuracy", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6, pad=1)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_calibration")

    # (c) 公開ベンチマークの概要
    bs = pd.read_parquet(REPO_ROOT / "data" / "benchmark" / "skus.parquet")
    bp = pd.read_parquet(REPO_ROOT / "data" / "benchmark" / "pairs.parquet")
    years = bs.groupby("release_year").size()
    rows = [f"SKUs (original designs) & {len(bs):,} " + r"\\",
            f"Designs (distinct artwork) & {bs.design_id.nunique():,} " + r"\\",
            "SKUs by release year & " + ", ".join(f"{y}: {n}" for y, n in years.items()) + r" \\",
            f"Evaluation pairs (same release month) & {len(bp):,} " + r"\\",
            r"Sales label & within-month decile of 8-week online sales \\",
            r"Pair label & which SKU sold more \\",
            r"Metadata & release month, price band (4 levels), image URL \\",
            r"Reference scores & Gemini Pro and Flash $\hat{p}$ for each pair \\"]
    (TAB / "tab_benchmark.tex").write_text("\n".join([r"\begin{tabular}{@{}p{0.36\linewidth}p{0.6\linewidth}@{}}", r"\toprule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n",
                                           encoding="utf-8")


def drivers(d: pd.DataFrame, n_bs: int, seed: int) -> None:
    """Gemini Pro の正解率が何に依存するか（3 パネル、横幅いっぱい）→ fig_drivers。decisions_log 2026-10-12（2）。

    (a) 確信度の高い順に判定する組を増やしたときの正解率、(b) 表明した確信度と正解率、(c) 組内の売上差の四分位別（探索的）。
    色は組の種類（青＝IP 内、オレンジ＝オリジナル同士）、灰色は B2（実線＝IP 内、破線＝オリジナル同士）。
    """
    rng = np.random.default_rng(seed)
    for m in ("pro_v0", "B2"):
        d[f"c_{m}"] = correct(d[m], d["truth"])
    d["conf"] = d["pro_v0"].abs() + 0.5
    d["gap"] = (np.log1p(d.first_y_raw.astype(float)) - np.log1p(d.second_y_raw.astype(float))).abs()
    col = {"ip": "C0", "original": "C1"}
    ls = {"ip": "-", "original": "--"}
    # 1 段幅の 2×2 配置（右下は凡例）
    fig, grid = plt.subplots(2, 2, figsize=(3.3, 2.7))
    axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
    grid[1, 1].axis("off")
    cov = np.linspace(0.1, 1.0, 19)

    # (a) coverage–accuracy
    ax = axes[0]
    for st in ("ip", "original"):
        g = d[d["set"] == st]
        for m, c, lw in (("pro_v0", col[st], 1.6), ("B2", "0.45", 1.0)):
            cc = g.loc[g[m].abs().sort_values(ascending=False, kind="stable").index, f"c_{m}"].to_numpy()
            acc = [cc[: max(1, int(round(k * len(cc))))].mean() for k in cov]
            ax.plot(100 * cov, 100 * np.array(acc), color=c, lw=lw, ls=ls[st] if m == "B2" else "-")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 83)

    # (b) calibration
    ax = axes[1]
    d["bin"] = pd.cut(d["conf"], [0.5, 0.55, 0.7, 0.775, 0.85, 1.0], include_lowest=True)
    for st, mk in (("ip", "o"), ("original", "s")):
        g = d[d["set"] == st]
        xs, ys, lo, hi = [], [], [], []
        for _, gb in g.groupby("bin", observed=True):
            if len(gb) < 20:
                continue
            t = boot(gb, ["pro_v0", "B2"], n_bs, rng).loc["pro_v0"]
            xs.append(gb.conf.mean()); ys.append(100 * t.acc); lo.append(100 * (t.acc - t.lo)); hi.append(100 * (t.hi - t.acc))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk + "-", color=col[st], ms=3, lw=1, capsize=1.5)
    ax.plot([0.5, 0.9], [50, 90], color="0.6", ls=":", lw=0.9)
    ax.text(0.495, 90, "dotted: calibrated", fontsize=5, color="0.4", ha="left", va="top")
    ax.set(xlim=(0.48, 0.92), ylim=(40, 92))
    ax.set_xlabel("Stated confidence")

    # (c) sales gap（探索的）
    ax = axes[2]
    for st, off in (("ip", -0.08), ("original", 0.08)):
        g = d[d["set"] == st].copy()
        g["q"] = pd.qcut(g.gap, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for m, c, mk, lsty in (("pro_v0", col[st], "o", "-"), ("B2", "0.45", ".", ls[st])):
            est = [boot(g[g.q == q], ["pro_v0", "B2"], n_bs, rng).loc[m] for q in ["Q1", "Q2", "Q3", "Q4"]]
            x = np.arange(4) + off
            ax.errorbar(x, [100 * e.acc for e in est], yerr=[[100 * (e.acc - e.lo) for e in est], [100 * (e.hi - e.acc) for e in est]],
                        fmt=mk, ls=lsty, color=c, ms=3, lw=1 if m == "pro_v0" else 0.8, capsize=1.2, elinewidth=0.7)
    ax.set(xticks=range(4), xticklabels=["Q1", "Q2", "Q3", "Q4"], ylim=(40, 85))
    ax.set_xlabel("Sales-gap quartile")

    for k, ax in enumerate(axes):
        ax.axhline(50, color="0.85", lw=0.7, ls="--", zorder=0)
        ax.set_title(f"({'abc'[k]})", fontsize=6.5, loc="left", pad=1)
        ax.tick_params(labelsize=5.5, pad=1, length=2)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(6)
        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="C0", lw=1.6, label="Gemini Pro, within-IP"),
               Line2D([], [], color="C1", lw=1.6, label="Gemini Pro, original-design"),
               Line2D([], [], color="0.45", lw=1.0, label="B2, within-IP"),
               Line2D([], [], color="0.45", lw=1.0, ls="--", label="B2, original-design")]
    grid[1, 1].legend(handles=handles, frameon=False, fontsize=5.5, loc="center", handlelength=1.8, labelspacing=0.6)
    axes[1].set_ylabel("Accuracy (%)")
    axes[2].set_ylabel("Accuracy (%)")
    fig.tight_layout(pad=0.2, w_pad=0.6, h_pad=0.5)
    save(fig, "fig_drivers")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    n_bs, seed = settings()
    stage1(n_bs, seed)
    stage1_decomposition()
    stage1_other_outputs(n_bs, seed)
    pairwise(load(), n_bs, seed)
    qualitative(load())
    exploratory(load(), n_bs, seed)
    extras(load(), n_bs, seed)
    drivers(load(), n_bs, seed)
    for p in sorted([*FIG.glob("*.pdf"), *FIG.glob("*.png"), *TAB.glob("*.tex")]):
        print(p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
