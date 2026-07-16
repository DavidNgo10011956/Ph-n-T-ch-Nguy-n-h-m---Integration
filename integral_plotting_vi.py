"""Matplotlib plotting for the integral analyser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib
# Forced to a non-interactive backend: this module now runs inside a
# headless Streamlit server (no display, no window manager), so an
# interactive backend like TkAgg would either fail to import or fail when
# plt.show() tries to open an actual window. plot_results() returns the
# Figure object instead; the caller (e.g. streamlit_app.py) is responsible
# for rendering it (st.pyplot(fig)) or saving it.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import Expr, Symbol

from integral_engine_vi import (
    constant_suffix,
    definite_integral,
    geometric_area,
    area_education_lines,
    first_integral_working,
    integral_label,
    readable_integral,
    special_function_note,
)
from integral_point_analysis_vi import InflectionPoint, SpecialPoint
from utils_vi import C_GREEN, C_RED, C_RESET, C_YELLOW, human_math, italicize_x, safe_eval, safe_lambdify


@dataclass(frozen=True)
class PlotConfig:
    n_plot_points: int = 2000
    figure_width: float = 19.0
    figure_height: float = 10.0
    y_pad_fraction: float = 0.15
    y_clip: float = 200.0
    colours: tuple[str, ...] = (
        "#2563EB",  # f(x)   — blue
        "#DC2626",  # F(x)   — red
        "#16A34A",  # F2(x)  — green
        "#9333EA",  # F3(x)  — purple
    )
    area_colour: str = "#FDBA74"


CFG = PlotConfig()

MARKER_STYLES: dict[str, tuple[str, str]] = {
    "Cực đại địa phương": ("^", "#DC2626"),
    "Cực tiểu địa phương": ("v", "#16A34A"),
    "Điểm dừng phẳng":    ("o", "#9333EA"),
    "Điểm tới hạn":       ("o", "#9333EA"),
    "Điểm uốn":           ("D", "#F97316"),
}

SHORT_MEANINGS: dict[int, str] = {
    1: "Diện tích tích lũy dưới f(x) (tính đến hằng số C)",
    2: "Nguyên hàm của F(x) — tích lũy bậc hai",
    3: "Nguyên hàm của F₂(x) — tích lũy bậc ba",
}

INTEGRAL_PLAIN_ENGLISH: dict[int, str] = {
    1: (
        "F′(x) = f(x)  →  F(x) là nguyên hàm của f(x)\n"
        "f(x) > 0  →  F(x) đang tăng (diện tích cộng thêm)\n"
        "f(x) < 0  →  F(x) đang giảm (diện tích trừ đi)\n"
        "f(x) = 0  →  F(x) có thể đạt cực đại hoặc cực tiểu"
    ),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_results(
    expr: Expr,
    integrals: dict[int, Expr],
    critical_points: list[SpecialPoint],
    inflection_points: list[InflectionPoint],
    x: Symbol,
    order: int,
    x_range: tuple[float, float],
    area_bounds: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
):
    """Build a Matplotlib Figure showing f(x), its antiderivative(s), special
    points and an explanation panel, and return it.

    Does not call plt.show() — this module runs headless. The caller
    decides how to display the returned Figure (e.g. st.pyplot(fig) in
    Streamlit) or save it themselves.
    """
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, CFG.n_plot_points)

    fig = plt.figure(figsize=(CFG.figure_width, CFG.figure_height), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.15], wspace=0.08)
    ax_graph = fig.add_subplot(grid[0, 0])
    ax_panel = fig.add_subplot(grid[0, 1])
    ax_panel.axis("off")

    y_arrays = _plot_all_curves(ax_graph, expr, integrals, x, x_values, order)
    if area_bounds:
        _shade_area(ax_graph, expr, x, area_bounds, x_min, x_max)
    y_min, y_max = _smart_ylim(y_arrays, critical_points, inflection_points)
    _format_graph(ax_graph, expr, x_min, x_max, y_min, y_max)
    _add_all_markers(ax_graph, critical_points, inflection_points, x_min, x_max, y_min, y_max)
    _build_explanation_panel(ax_panel, expr, integrals, critical_points, inflection_points, x, order, area_bounds)

    if save_path:
        _save_figure(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Curve plotting
# ---------------------------------------------------------------------------

def _plot_all_curves(ax_graph, expr: Expr, integrals: dict[int, Expr], x: Symbol, x_values, order: int):
    y_arrays = []
    _plot_curve(ax_graph, expr, x, x_values, "f(x)", CFG.colours[0], 2.5, "-", y_arrays)
    for n in range(1, min(order, 3) + 1):
        _plot_curve(
            ax_graph, integrals[n], x, x_values,
            integral_label(n), CFG.colours[n], 1.8,
            "--" if n > 1 else "-", y_arrays,
        )
    return y_arrays


def _plot_curve(ax_graph, sym_expr: Expr, x: Symbol, x_values, label: str, colour: str,
                line_width: float, line_style: str, y_arrays: list) -> None:
    try:
        numeric_func = safe_lambdify(x, sym_expr)
        y_values = safe_eval(numeric_func, x_values)
        y_arrays.append(y_values)
        ax_graph.plot(x_values, y_values, color=colour, lw=line_width, linestyle=line_style, label=italicize_x(label))
    except Exception as err:
        print(f"{C_YELLOW}Không thể vẽ {label}: {err}{C_RESET}")
        y_arrays.append(None)


def _shade_area(ax_graph, expr: Expr, x: Symbol, area_bounds: tuple[float, float], x_min: float, x_max: float) -> None:
    a, b = area_bounds
    a, b = max(a, x_min), min(b, x_max)
    if a >= b:
        return
    try:
        numeric_func = safe_lambdify(x, expr)
        shade_x = np.linspace(a, b, 800)
        shade_y = safe_eval(numeric_func, shade_x)
        ax_graph.fill_between(shade_x, 0, shade_y, color=CFG.area_colour, alpha=0.45,
                              label=italicize_x(f"Diện tích [{a:.3g}, {b:.3g}]"), zorder=1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Graph formatting
# ---------------------------------------------------------------------------

def _format_graph(ax_graph, expr: Expr, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    ax_graph.set_xlim(x_min, x_max)
    ax_graph.set_ylim(y_min, y_max)
    ax_graph.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax_graph.axvline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax_graph.set_title(italicize_x(f"Hàm số và nguyên hàm\nf(x) = {human_math(expr)}"), fontsize=15, pad=10)
    ax_graph.set_xlabel(italicize_x("x"), fontsize=11)
    ax_graph.set_ylabel("y", fontsize=11)
    ax_graph.grid(True, alpha=0.30, linestyle=":")
    ax_graph.legend(fontsize=9, loc="upper right", framealpha=0.9)


def _smart_ylim(y_arrays, critical_points: list[SpecialPoint], inflection_points: list[InflectionPoint]) -> tuple[float, float]:
    parts = [a[np.isfinite(a)] for a in y_arrays if a is not None and np.any(np.isfinite(a))]
    special_y = [y for _, y, _ in critical_points] + [y for _, y in inflection_points]
    if special_y:
        parts.append(np.array([v for v in special_y if np.isfinite(v)]))
    if not parts:
        return -10.0, 10.0
    data = np.concatenate(parts)
    y_min, y_max = np.nanpercentile(data, [2.0, 98.0])
    if special_y:
        finite_special = [v for v in special_y if np.isfinite(v)]
        y_min = min(y_min, min(finite_special))
        y_max = max(y_max, max(finite_special))
    if abs(y_max - y_min) < 1e-8:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    pad = CFG.y_pad_fraction * (y_max - y_min)
    return max(y_min - pad, -CFG.y_clip), min(y_max + pad, CFG.y_clip)


# ---------------------------------------------------------------------------
# Markers for special points (critical / inflection points OF F(x))
# ---------------------------------------------------------------------------

def _add_all_markers(ax_graph, critical_points, inflection_points, x_min, x_max, y_min, y_max) -> None:
    used_labels: set[str] = set()
    for x_value, y_value, kind in critical_points:
        _add_marker(ax_graph, x_value, y_value, kind, used_labels, x_min, x_max, y_min, y_max)
    for x_value, y_value in inflection_points:
        _add_marker(ax_graph, x_value, y_value, "Điểm uốn", used_labels, x_min, x_max, y_min, y_max)


def _add_marker(ax_graph, x_value: float, y_value: float, kind: str, used_labels: set[str],
                x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    if not (x_min <= x_value <= x_max and y_min <= y_value <= y_max):
        return
    marker, colour = MARKER_STYLES.get(kind, ("o", "#9333EA"))
    label = f"F(x): {kind}" if kind not in used_labels else None
    used_labels.add(kind)
    ax_graph.scatter(x_value, y_value, marker=marker, color=colour, s=90,
                     edgecolors="white", linewidths=0.8, zorder=6, label=label)
    offset = 0.045 * (y_max - y_min)
    ax_graph.annotate(
        f"{kind}\n({x_value:.3f}, {y_value:.3f})",
        xy=(x_value, y_value), xytext=(x_value, y_value + offset),
        fontsize=7.8, ha="center", va="bottom", color=colour, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=colour,
                  linewidth=0.9, alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=colour, lw=0.9, alpha=0.7),
    )


# ---------------------------------------------------------------------------
# Explanation panel
# ---------------------------------------------------------------------------

def _build_explanation_panel(ax_panel, expr, integrals, critical_points, inflection_points, x, order,
                              area_bounds) -> None:
    ax_panel.set_title("GIẢI THÍCH DÀNH CHO HỌC SINH", fontsize=15, pad=10)

    _panel(ax_panel, 0.98, "① CÁC BƯỚC TÍNH NGUYÊN HÀM",
           _integral_panel_lines(expr, integrals, x, order), font_size=8.9)

    _panel(ax_panel, 0.40, "② KẾT QUẢ DIỆN TÍCH",
           _area_panel_lines(expr, x, area_bounds), font_size=9.0)

    _panel(ax_panel, 0.25, "③ ĐIỂM ĐẶC BIỆT CỦA F(x)",
           _special_point_panel_lines(critical_points, inflection_points), font_size=8.8)

    _panel(ax_panel, 0.10, "④ CÁCH ĐỌC ĐỒ THỊ",
           _meaning_panel_lines(order), font_size=8.8)

    row = -0.04
    note = special_function_note(str(expr))
    if note:
        _panel(ax_panel, row, "⑤ THÔNG TIN VỀ HÀM SỐ", [note], font_size=9.0)


def _integral_panel_lines(expr, integrals, x, order) -> list[str]:
    lines = [f"f(x) = {human_math(expr)}"]
    lines.extend(first_integral_working(expr, integrals[1], x))
    lines.append("")
    lines.append("Tóm tắt nguyên hàm:")
    for n in range(1, min(order, 3) + 1):
        disp = readable_integral(integrals[n], n)
        lines.append(f"  {integral_label(n)} = {human_math(disp)}  {constant_suffix(n)}")
    return lines



def _area_panel_lines(expr, x, area_bounds) -> list[str]:
    if not area_bounds:
        return ["Chưa chọn cận tích phân xác định."]

    a, b = area_bounds
    signed = definite_integral(expr, x, a, b)
    geometric = geometric_area(expr, x, a, b)

    lines = [f"Khoảng tính: [{a:g}, {b:g}]"]
    if signed is None:
        lines.append("Diện tích có dấu: không tính được")
    else:
        lines.append(f"Diện tích có dấu  ∫f(x)dx   = {signed:.6f}")

    if geometric is None:
        lines.append("Diện tích hình học: không tính được")
    else:
        lines.append(f"Diện tích hình học ∫|f(x)|dx = {geometric:.6f}")

    if signed is not None and geometric is not None:
        lines.extend([
            "Trên Ox: cộng (+); dưới Ox: trừ (−).",
            "Với ∫|f(x)|dx, mọi phần diện tích đều dương.",
        ])
    else:
        lines.append("")
        lines.append("VÌ SAO KHÔNG TÍNH ĐƯỢC?")
        lines.extend(f"• {text}" for text in area_education_lines(expr, x, a, b))

    # Giữ bảng gọn để không chồng lên các mục bên dưới.
    return lines[:10]

def _special_point_panel_lines(critical_points, inflection_points) -> list[str]:
    total = len(critical_points) + len(inflection_points)
    lines = [f"Tổng số điểm đặc biệt của F(x): {total}",
             "(tại các điểm f(x) = 0, hoặc f′(x) = 0 đổi dấu)"]
    if critical_points:
        for x_value, y_value, kind in critical_points:
            lines.append(f"  {kind}: x = {x_value:.4g},  F(x) = {y_value:.4g}")
    else:
        lines.append("  Cực trị của F: không có trong khoảng này")
    if inflection_points:
        for x_value, y_value in inflection_points:
            lines.append(f"  Điểm uốn của F: x = {x_value:.4g},  F(x) = {y_value:.4g}")
    else:
        lines.append("  Điểm uốn của F: không có trong khoảng này")
    if len(lines) > 10:
        return lines[:10] + ["  … more points hidden for readability"]
    return lines


def _meaning_panel_lines(order: int) -> list[str]:
    lines: list[str] = []
    for n in range(1, min(order, 3) + 1):
        lines.append(f"{integral_label(n)}: {SHORT_MEANINGS.get(n, 'Nguyên hàm bậc cao')}")
    lines.append("")
    if 1 in INTEGRAL_PLAIN_ENGLISH:
        lines.append(INTEGRAL_PLAIN_ENGLISH[1])
        lines.append("")
    lines.append("Cực đại / cực tiểu của F: f(x) đổi dấu tại đây")
    lines.append("Điểm uốn của F: f′(x) đổi dấu tại đây")
    return lines


def _panel(ax_panel, y_top: float, title: str, body: list[str], font_size: float) -> None:
    ax_panel.text(
        0.02, y_top, italicize_x(title),
        transform=ax_panel.transAxes, fontsize=12.5, fontweight="bold", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E0F2FE", edgecolor="#0369A1", linewidth=1.1),
    )
    ax_panel.text(
        0.02, y_top - 0.055, italicize_x("\n".join(body)),
        transform=ax_panel.transAxes, fontsize=font_size, va="top", ha="left",
        family="DejaVu Sans", linespacing=1.08,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.0),
    )


def _save_figure(fig, save_path: str) -> None:
    try:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"{C_GREEN}Đã lưu hình → {save_path}{C_RESET}")
    except Exception as err:
        print(f"{C_RED}Không thể lưu hình: {err}{C_RESET}")
