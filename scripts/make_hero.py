"""論文の全体像の図（ティーザー）→ paper/figures/fig_hero.{pdf,png} と部品 paper/figures/hero_parts/*.{svg,png}。

`make figures` から呼ぶ。イラスト（人物・AI・吹き出し・判定マーク）は図形の組み合わせで描く。
部品は背景透明で 1 つずつ書き出し、Google スライド等で並べ直せるようにする。
T シャツはオリジナル同士の組の写真だけを使い（select_qualitative の assert で確認）、投稿版なのでロゴを伏せる（mask_logo）。
数値は集計し直して入れる（手で書かない）。配色は dataviz の基準パレットの青・オレンジと、判定マークの緑・赤（記号と文字を必ず併記）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_pairwise_robustness import correct, load  # noqa: E402
from scripts.make_figures import mask_logo, select_qualitative  # noqa: E402
from src.eval.metrics import spearman  # noqa: E402
from src.ingest.bq import REPO_ROOT  # noqa: E402

FIG = REPO_ROOT / "paper" / "figures"
PARTS = FIG / "hero_parts"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1f8a3b", "#d03b3b"
INK, INK2, PANEL, LINE = "#0b0b0b", "#52514e", "#f4f3f0", "#c9c7c1"
SKIN = ["#f2c9a5", "#d9a77c", "#a8754f", "#f6d7bd", "#c58c62"]
SHIRT = [BLUE, ORANGE, "#1baf7a", "#4a3aa7", "#e87ba4"]


# ---------------------------------------------------------------- イラスト部品（データ座標で描く）
def person(ax, x: float, y: float, s: float = 1.0, shirt: str = BLUE, skin: str = SKIN[0], hair: str = "#3b2f2a") -> None:
    """立ち姿の人物（上半身）。(x, y) は足元の中心。"""
    ax.add_patch(FancyBboxPatch((x - 0.9 * s, y), 1.8 * s, 1.5 * s, boxstyle=f"round,pad=0,rounding_size={0.55 * s}",
                                fc=shirt, ec="none", zorder=3))
    ax.add_patch(Circle((x, y + 2.05 * s), 0.62 * s, fc=skin, ec="none", zorder=4))
    ax.add_patch(Wedge((x, y + 2.12 * s), 0.68 * s, 8, 172, fc=hair, ec="none", zorder=5))


def robot(ax, x: float, y: float, s: float = 1.0, color: str = BLUE) -> None:
    """AI（視覚言語モデル）のアイコン。(x, y) は中心。"""
    ax.add_patch(FancyBboxPatch((x - 1.3 * s, y - 1.0 * s), 2.6 * s, 2.0 * s, boxstyle=f"round,pad=0,rounding_size={0.45 * s}",
                                fc=color, ec="none", zorder=3))
    for dx in (-0.55, 0.55):
        ax.add_patch(Circle((x + dx * s, y + 0.15 * s), 0.3 * s, fc="white", ec="none", zorder=4))
        ax.add_patch(Circle((x + dx * s, y + 0.15 * s), 0.13 * s, fc=INK, ec="none", zorder=5))
    ax.add_patch(Rectangle((x - 0.5 * s, y - 0.6 * s), 1.0 * s, 0.16 * s, fc="white", ec="none", zorder=4))
    ax.plot([x, x], [y + 1.0 * s, y + 1.45 * s], color=color, lw=2, zorder=2)
    ax.add_patch(Circle((x, y + 1.55 * s), 0.18 * s, fc=color, ec="none", zorder=3))
    for dx in (-1.3, 1.3):
        ax.add_patch(FancyBboxPatch((x + dx * s - 0.18 * s, y - 0.35 * s), 0.36 * s, 0.7 * s,
                                    boxstyle=f"round,pad=0,rounding_size={0.15 * s}", fc=color, ec="none", zorder=2))


def bubble(ax, x: float, y: float, w: float, h: float, text: str, tail: tuple[float, float], fs: float = 7.5) -> None:
    """吹き出し。(x, y) は左下、tail は尾の先端。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=LINE, lw=0.8, zorder=6))
    bx = min(max(tail[0], x + 0.8), x + w - 0.8)
    by = y if tail[1] < y else y + h
    ax.add_patch(Polygon([(bx - 0.5, by), (bx + 0.5, by), tail], fc="white", ec=LINE, lw=0.8, zorder=5))
    ax.plot([bx - 0.45, bx + 0.45], [by, by], color="white", lw=1.6, zorder=7)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=8)


def verdict(ax, x: float, y: float, ok: bool, s: float = 1.0) -> None:
    """判定マーク（緑のチェック／赤のバツ）。色だけに頼らず、記号の形でも区別する。"""
    ax.add_patch(Circle((x, y), 0.75 * s, fc=GREEN if ok else RED, ec="none", zorder=6))
    if ok:
        ax.plot([x - 0.38 * s, x - 0.08 * s, x + 0.42 * s], [y, y - 0.32 * s, y + 0.34 * s], color="white", lw=2.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=7)
    else:
        for sgn in (1, -1):
            ax.plot([x - 0.33 * s, x + 0.33 * s], [y - 0.33 * s * sgn, y + 0.33 * s * sgn], color="white", lw=2.4,
                    solid_capstyle="round", zorder=7)


def arrow(ax, p0: tuple[float, float], p1: tuple[float, float], color: str = INK2) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.3, color=color, zorder=2))


def shirt(ax, img: Image.Image, x: float, y: float, h: float, label: str | None = None) -> None:
    """T シャツの写真。(x, y) は左下、h は高さ（データ座標）。"""
    w = h * img.width / img.height
    ax.imshow(img, extent=(x, x + w, y, y + h), zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=LINE, lw=0.6, zorder=4))
    if label:
        ax.add_patch(Circle((x + 0.55, y + h - 0.55), 0.45, fc=INK, ec="none", zorder=5))
        ax.text(x + 0.55, y + h - 0.57, label, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=6)


# ---------------------------------------------------------------- 数値と画像
def inputs() -> dict:
    d = load()
    for m in ("pro_v0", "B2"):
        d[f"c_{m}"] = correct(d[m], d["truth"])
    ip = d[d["set"] == "ip"]
    top = ip[ip["pro_v0"].abs().rank(pct=True, method="first") > 0.75]
    proc = REPO_ROOT / "data" / "processed"
    base = pd.read_parquet(proc / "baseline_predictions.parquet")
    sc = pd.read_parquet(proc / "panel" / "stage1_scores.parquet").query("condition == 'full'")
    s1 = base.merge(sc[["class_id", "color_id", "intent_mean"]], on=["class_id", "color_id"])
    sel = select_qualitative(d)
    man = pd.read_parquet(REPO_ROOT / "data" / "cache" / "image_manifest.parquet").set_index(["class_id", "color_id"])
    img = lambda r, side: mask_logo(Image.open(REPO_ROOT / man.loc[(r[f"{side}_class_id"], r[f"{side}_color_id"]), "image_path"]))  # noqa: E731
    a, c = sel.iloc[0], sel.iloc[2]
    return {"acc": ip["c_pro_v0"].mean(), "b2": ip["c_B2"].mean(), "top": top["c_pro_v0"].mean(),
            "rho": spearman(s1.y_raw.to_numpy(float), s1.intent_mean.to_numpy()),
            "n_sku": 2304, "img_single": img(c, "second"), "img_a": img(a, "first"), "img_b": img(a, "second")}


def model_box(ax, x: float, y: float, w: float, h: float, title: str, sub: str) -> None:
    """落ち着いた版で人物・ロボットの代わりに使う、AI の役割を書いた箱。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc="white", ec=BLUE, lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7, fontweight="bold", color=INK, zorder=4,
            linespacing=1.15)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=6, color=INK2, zorder=4, linespacing=1.2)


def prompt_box(ax, x: float, y: float, w: float, h: float, text: str) -> None:
    """落ち着いた版で吹き出しの代わりに使う、実際の質問を書いた箱。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc="#e8f1fc", ec="none", zorder=3))
    ax.text(x + 0.5, y + h - 0.45, "prompt", ha="left", va="top", fontsize=5.5, color=INK2, style="italic", zorder=4)
    ax.text(x + w / 2, y + h * 0.42, text, ha="center", va="center", fontsize=6.6, color=INK, zorder=4, linespacing=1.2)


# ---------------------------------------------------------------- 全体図
def hero(style: str = "clean") -> None:
    """style="clean"（論文の標準：箱と矢印だけ）／"illustrated"（人物・ロボットのイラスト入り、fig_hero_illustrated）。"""
    v = inputs()
    fig = plt.figure(figsize=(7.0, 3.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 70), ylim=(0, 30), aspect="equal")
    ax.axis("off")

    # 上段の 2 枚のパネル
    for x0, title, col in ((0.5, "Study 1: rate one product at a time", RED), (35.5, "Study 2: compare two designs within a group", GREEN)):
        ax.add_patch(FancyBboxPatch((x0, 7.6), 34, 21.9, boxstyle="round,pad=0,rounding_size=1.2", fc=PANEL, ec="none", zorder=0))
        ax.text(x0 + 1.2, 27.6, title, fontsize=9, fontweight="bold", color=INK, va="center")
    ax.text(35, 29.6, "Same model, same products — only the question changes", ha="center", va="top", fontsize=7, color=INK2,
            style="italic", bbox=dict(fc="white", ec="none", pad=1.5), zorder=9)

    # Study 1：1 枚の T シャツ → 合成消費者 5 人 → 相関ほぼ 0
    if style == "illustrated":
        shirt(ax, v["img_single"], 2.0, 13.4, 9.6)
        arrow(ax, (9.6, 18.2), (12.0, 18.2))
        for i, (px, py) in enumerate(((14.0, 15.6), (17.0, 15.6), (20.0, 15.6), (15.5, 12.2), (18.5, 12.2))):
            person(ax, px, py, 0.95, SHIRT[i], SKIN[i])
        robot(ax, 24.0, 14.6, 0.75)
        ax.text(24.0, 12.3, "played by\na VLM", ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.2)
        ax.text(16.6, 12.3, "50 synthetic\nconsumers", ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.2)
        bubble(ax, 12.8, 21.0, 12.4, 3.8, "How much do you want\nto buy it?  (1–5)", (16.8, 19.9), fs=6.8)
        arrow(ax, (26.3, 18.2), (28.6, 18.2))
    else:
        shirt(ax, v["img_single"], 2.4, 13.6, 8.4)
        arrow(ax, (8.8, 17.8), (11.4, 17.8))
        prompt_box(ax, 11.8, 20.4, 14.0, 4.6, "\u201cHow much do you want to buy\nthis T-shirt?\u201d  (1\u20135)")
        model_box(ax, 11.8, 13.0, 14.0, 6.6, "VLM playing 50\nsynthetic consumers", "personas matched to the\nretailer's customer base")
        arrow(ax, (26.2, 18.9), (28.8, 18.9))
    verdict(ax, 31.4, 22.0, ok=False)
    ax.text(31.4, 18.6, f"$\\rho \\approx {abs(v['rho']):.0f}$" if abs(v["rho"]) < 0.005 else f"$\\rho = {v['rho']:.2f}$",
            ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(31.4, 16.6, "correlation\nwith sales", ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.2)
    ax.text(17.5, 8.6, "Misjudges franchise strength: rates originals as high as licensed designs",
            ha="center", fontsize=6.3, color=INK2)

    # Study 2：A/B の 2 枚 → ベテラン企画担当役の AI → 正解率
    if style == "illustrated":
        shirt(ax, v["img_a"], 37.0, 13.4, 9.6, "A")
        shirt(ax, v["img_b"], 44.2, 13.4, 9.6, "B")
        ax.text(44.6, 11.6, "same franchise / same month", ha="center", fontsize=6.5, color=INK2)
    else:
        shirt(ax, v["img_a"], 36.8, 13.6, 8.4, "A")
        shirt(ax, v["img_b"], 43.2, 13.6, 8.4, "B")
        ax.text(42.9, 12.0, "same franchise /\nsame month", ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.2)
    if style == "illustrated":
        arrow(ax, (51.8, 18.2), (53.2, 18.2))
        person(ax, 55.2, 14.2, 1.1, "#3f3d39", SKIN[1])
        robot(ax, 58.6, 15.6, 0.7)
        bubble(ax, 51.6, 21.0, 10.0, 3.8, "Which one sold more\nonline in 8 weeks?", (55.2, 18.4), fs=6.8)
        ax.text(56.9, 12.3, "merchandiser role\nplayed by a VLM", ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.2)
        arrow(ax, (60.4, 18.2), (62.2, 18.2))
    else:
        arrow(ax, (49.4, 17.8), (51.0, 17.8))
        prompt_box(ax, 51.3, 20.4, 9.6, 4.6, "\u201cWhich one sold more\nonline in 8 weeks?\u201d")
        model_box(ax, 51.3, 13.0, 9.6, 6.6, "VLM playing\na merchandiser", "sees both images,\nprices, franchise")
        arrow(ax, (61.2, 18.9), (62.8, 18.9))
    rx = 65.8 if style == "illustrated" else 66.3
    verdict(ax, rx, 22.0, ok=True)
    ax.text(rx, 18.6, f"{100 * v['acc']:.1f}%", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(rx, 16.6, "accuracy", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.text(rx, 14.6, f"baseline {100 * v['b2']:.1f}%\nchance 50%", ha="center", va="top", fontsize=6.0, color=INK2, linespacing=1.3)
    ax.text(52.5, 8.6, f"About {100 * v['top']:.0f}% on the quarter of pairs the model is most confident about",
            ha="center", fontsize=6.3, color=INK2)

    # 下段：答え合わせ（実売上と期間）
    ax.text(0.8, 5.6, f"Validated against 8-week online sales of {v['n_sku']:,} graphic T-shirts", fontsize=7.5, fontweight="bold",
            color=INK, va="center")
    segs = (("2023–2024: train baselines", 0.5, 22.5, "#d6d4cf"), ("2025: test", 23.0, 13.5, "#b7d3f6"),
            ("2026: preregistered confirmation", 36.5, 33.0, "#86b6ef"))
    for lab, x, w, c in segs:
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 3.0, boxstyle="round,pad=0,rounding_size=0.6", fc=c, ec="none", zorder=1))
        ax.text(x + w / 2, 2.5, lab, ha="center", va="center", fontsize=7, color=INK)

    name = "fig_hero" if style == "clean" else "fig_hero_illustrated"
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- 部品（背景透明、1 つずつ）
def parts() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)

    def save(name: str, draw, xlim: tuple[float, float], ylim: tuple[float, float], size: tuple[float, float]) -> None:
        fig = plt.figure(figsize=size)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=xlim, ylim=ylim, aspect="equal")
        ax.axis("off")
        draw(ax)
        for ext in ("svg", "png"):
            fig.savefig(PARTS / f"{name}.{ext}", dpi=300, transparent=True)
        plt.close(fig)

    for i, c in enumerate(SHIRT):
        save(f"person_{i + 1}", lambda ax, i=i, c=c: person(ax, 1.2, 0.1, 1.0, c, SKIN[i]), (0, 2.4), (0, 2.9), (1.2, 1.45))
    save("person_merchandiser", lambda ax: person(ax, 1.2, 0.1, 1.0, "#3f3d39", SKIN[1]), (0, 2.4), (0, 2.9), (1.2, 1.45))
    save("crowd_5", lambda ax: [person(ax, px, py, 0.95, SHIRT[i], SKIN[i]) for i, (px, py) in
                                enumerate(((1.3, 3.5), (4.3, 3.5), (7.3, 3.5), (2.8, 0.1), (5.8, 0.1)))], (0, 8.6), (0, 6.3), (2.6, 1.9))
    save("ai_robot", lambda ax: robot(ax, 1.6, 1.2, 1.0), (0, 3.2), (0, 3.0), (1.6, 1.5))
    save("check_ok", lambda ax: verdict(ax, 0.8, 0.8, True), (0, 1.6), (0, 1.6), (0.8, 0.8))
    save("cross_ng", lambda ax: verdict(ax, 0.8, 0.8, False), (0, 1.6), (0, 1.6), (0.8, 0.8))
    save("bubble_rating", lambda ax: bubble(ax, 0.2, 1.2, 12.4, 3.8, "How much do you want\nto buy it?  (1–5)", (4.0, 0.1)),
         (0, 12.8), (0, 5.2), (3.2, 1.3))
    save("bubble_pairwise", lambda ax: bubble(ax, 0.2, 1.2, 10.6, 3.8, "Which one sold more\nonline in 8 weeks?", (3.6, 0.1)),
         (0, 11.0), (0, 5.2), (2.75, 1.3))
    v = inputs()
    for name, key in (("tshirt_single", "img_single"), ("tshirt_A", "img_a"), ("tshirt_B", "img_b")):
        v[key].save(PARTS / f"{name}.png")


def main() -> None:
    hero("clean")
    hero("illustrated")
    parts()


if __name__ == "__main__":
    main()
