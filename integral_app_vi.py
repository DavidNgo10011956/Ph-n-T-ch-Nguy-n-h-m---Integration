# Author: David Ngo (derivative version) — converted to integration by Claude
# Purpose : Lấy nguyên hàm (tích phân bất định) và tích phân xác định của các
# phương trình toán học dùng Python. Đây là phiên bản "đảo ngược" của bộ công
# cụ đạo hàm gốc, dùng chung phong cách trình bày và cùng bộ 7 tệp:
# integral_app_vi.py, integral_engine_vi.py, Human_readable_format_vi.py,
# integral_input_guide_vi.py, integral_plotting_vi.py, integral_point_analysis_vi.py,
# và utils_vi.py.

"""Ứng dụng dòng lệnh chính cho Chương trình Phân tích Tích phân Chuyên nghiệp."""

from __future__ import annotations

import sys
import time
from typing import Optional

from sympy import symbols

from integral_engine_vi import (
    compute_integrals,
    area_working_lines,
    constant_suffix,
    definite_integral,
    geometric_area,
    area_education_lines,
    integral_label,
    integral_title,
    parse_function,
    readable_integral,
    special_function_note,
)
from integral_input_guide_vi import banner, hr, is_special_function
from integral_plotting_vi import plot_results
from integral_point_analysis_vi import (
    InflectionPoint,
    SpecialPoint,
    find_special_points_of_antiderivative,
)
from utils_vi import (
    C_BOLD, C_CYAN, C_GREEN, C_MAGENTA, C_RESET, C_YELLOW,
    ask_float, ask_int, ask_yes_no, human_math,
)


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------

def analyse(
    function_text: str,
    order: int = 1,
    x_range: tuple[float, float] = (-6.0, 6.0),
    area_bounds: Optional[tuple[float, float]] = None,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> dict[str, object]:
    """Programmatic entry point — useful when importing from another script."""
    x = symbols("x")
    expr = parse_function(function_text, x)
    if expr is None:
        raise ValueError(f"Could not parse function: {function_text!r}")
    start = time.perf_counter()
    integrals = compute_integrals(expr, x, order)
    critical_points, inflection_points = find_special_points_of_antiderivative(
        expr, integrals[1], x, *x_range
    )
    area_value = definite_integral(expr, x, *area_bounds) if area_bounds else None
    geometric_area_value = geometric_area(expr, x, *area_bounds) if area_bounds else None
    elapsed = time.perf_counter() - start
    print_results(
        expr, integrals, critical_points, inflection_points,
        area_bounds, area_value, geometric_area_value, elapsed,
    )
    if plot:
        plot_results(
            expr, integrals, critical_points, inflection_points,
            x, order, x_range, area_bounds, save_path,
        )
    return {
        "expr": expr,
        "integrals": integrals,
        "critical_points": critical_points,
        "inflection_points": inflection_points,
        "area_bounds": area_bounds,
        "area_value": area_value,
        "signed_area": area_value,
        "geometric_area": geometric_area_value,
        "elapsed_time": elapsed,
    }


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_results(
    expr,
    integrals,
    critical_points: list[SpecialPoint],
    inflection_points: list[InflectionPoint],
    area_bounds: Optional[tuple[float, float]],
    area_value: Optional[float],
    geometric_area_value: Optional[float],
    elapsed: float,
) -> None:
    """Print a formatted summary to the terminal."""
    print("\n" + hr())

    # --- Function ---
    print(f"\n{C_CYAN}{C_BOLD}HÀM SỐ{C_RESET}")
    print("-" * 88)
    print(f"  f(x) = {C_BOLD}{human_math(expr)}{C_RESET}")

    note = special_function_note(str(expr))
    if note:
        print(f"\n  {C_MAGENTA}ℹ  {note}{C_RESET}")

    # --- Antiderivatives ---
    for order, integral_expr in integrals.items():
        print(f"\n{C_CYAN}{C_BOLD}{integral_title(order)}{C_RESET}")
        print("-" * 88)
        display_integral = readable_integral(integral_expr, order)
        print(f"  {integral_label(order)} = {C_BOLD}{human_math(display_integral)}  {constant_suffix(order)}{C_RESET}")

    # --- Definite integral: signed area and geometric area ---
    if area_bounds:
        a, b = area_bounds
        print(f"\n{C_CYAN}{C_BOLD}KẾT QUẢ TÍCH PHÂN XÁC ĐỊNH VÀ DIỆN TÍCH{C_RESET}")
        print("-" * 88)

        if area_value is None:
            print(f"  {C_YELLOW}Không thể tính tích phân xác định trên [{a}, {b}].{C_RESET}")
        else:
            print(f"  {'DIỆN TÍCH CÓ DẤU':<28} ∫[{a}, {b}] f(x) dx   =  "
                  f"{C_BOLD}{area_value:.6f}{C_RESET}")

        if geometric_area_value is None:
            print(f"  {C_YELLOW}Không thể tính diện tích hình học trên [{a}, {b}].{C_RESET}")
        else:
            print(f"  {'DIỆN TÍCH HÌNH HỌC':<28} ∫[{a}, {b}] |f(x)| dx =  "
                  f"{C_BOLD}{geometric_area_value:.6f}{C_RESET}")

        print(f"\n  {C_YELLOW}{C_BOLD}CÁC BƯỚC TÍNH CHI TIẾT:{C_RESET}")
        for line in area_working_lines(expr, symbols("x"), a, b):
            print(f"  {line}")


    # --- Special points of F(x) ---
    print(f"\n{C_CYAN}{C_BOLD}CÁC ĐIỂM ĐẶC BIỆT CỦA F(x){C_RESET}")
    print("-" * 88)
    total = len(critical_points) + len(inflection_points)
    print(f"  Tổng số điểm đặc biệt khác nhau tìm được: {C_BOLD}{total}{C_RESET}")
    print(f"  (Cực trị của F tại nơi f(x) = 0 đổi dấu; điểm uốn của F tại nơi f′(x) = 0 đổi dấu)")

    if critical_points:
        for x_value, y_value, kind in critical_points:
            print(f"  {kind:<24} x = {x_value: .6f},   F(x) = {y_value: .6f}")
    else:
        print(f"  {C_YELLOW}Không tìm thấy cực trị của F(x) trong khoảng x này.{C_RESET}")

    if inflection_points:
        for x_value, y_value in inflection_points:
            print(f"  {'Điểm uốn':<24} x = {x_value: .6f},   F(x) = {y_value: .6f}")
    else:
        print(f"  {C_YELLOW}Không tìm thấy điểm uốn của F(x) trong khoảng x này.{C_RESET}")

    # --- Timing ---
    print(f"\n{C_CYAN}{C_BOLD}THỜI GIAN XỬ LÝ{C_RESET}")
    print("-" * 88)
    print(f"  {elapsed:.4f} seconds")
    print(hr() + "\n")


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt_function(show_special: bool = False):
    """Ask user for f(x), with help / special / quit support."""
    x = symbols("x")
    while True:
        try:
            raw = input(f"{C_GREEN}Nhập f(x): {C_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã hủy.")
            sys.exit(0)

        lower = raw.lower()

        if lower in {"quit", "exit"}:
            print(f"\n{C_CYAN}Tạm biệt!{C_RESET}\n")
            sys.exit(0)

        if lower == "help":
            print(banner(show_special=show_special))
            continue

        if lower == "special":
            from integral_input_guide_vi import special_function_guide
            print(special_function_guide())
            continue

        if not raw:
            print(f"{C_YELLOW}Vui lòng nhập một biểu thức. Gõ 'help' để xem ví dụ.{C_RESET}")
            continue

        expr = parse_function(raw, x)
        if expr is not None:
            note = is_special_function(raw)
            if note:
                print(f"\n  {C_MAGENTA}ℹ  {note}{C_RESET}\n")
            return expr, raw

        print(f"{C_YELLOW}Không thể phân tích biểu thức này.{C_RESET}")
        print(f"{C_YELLOW}Ví dụs: x**3 - 2*x + 1,  sin(x**2),  x*sin(x),  exp(-x**2){C_RESET}")
        print(f"{C_YELLOW}Gõ 'help' để xem hướng dẫn đầy đủ.{C_RESET}")


def _prompt_x_range() -> tuple[float, float]:
    """Ask for x_min / x_max with validation."""
    while True:
        x_min = ask_float("Giới hạn trái x_min của đồ thị", default=-6.0)
        x_max = ask_float("Giới hạn phải x_max của đồ thị", default=6.0)
        if x_min < x_max:
            return x_min, x_max
        print(f"{C_YELLOW}x_min phải nhỏ hơn x_max. Vui lòng thử lại.{C_RESET}")


def _prompt_area_bounds(x_min: float, x_max: float) -> Optional[tuple[float, float]]:
    """Ask whether the user wants a definite integral, and over what bounds."""
    if not ask_yes_no("Tính tích phân xác định (diện tích) trên một khoảng?", default=False):
        return None
    while True:
        a = ask_float("Cận dưới a", default=x_min)
        b = ask_float("Cận trên b", default=x_max)
        if a < b:
            return a, b
        print(f"{C_YELLOW}Cận dưới a phải nhỏ hơn cận trên b. Vui lòng thử lại.{C_RESET}")


# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

def run_session() -> None:
    """Run the interactive terminal session."""
    x = symbols("x")

    print(banner(show_special=False))
    print(f"{C_CYAN}Mẹo: gõ 'special' bất cứ lúc nào để xem hướng dẫn về các hàm nâng cao.{C_RESET}\n")

    show_special = False

    while True:
        expr, raw = prompt_function(show_special=show_special)

        if is_special_function(raw):
            show_special = True

        order = ask_int(
            "Tính nguyên hàm lặp đến bậc nào? (1 = ∫f dx, 2 = ∫∫f dx dx, 3 = ∫∫∫f dx dx dx)",
            default=1, minimum=1, maximum=3,
        )
        x_min, x_max = _prompt_x_range()
        area_bounds = _prompt_area_bounds(x_min, x_max)

        print(f"\n{C_CYAN}Đang tính toán …{C_RESET}\n")
        start = time.perf_counter()
        integrals = compute_integrals(expr, x, order)
        critical_points, inflection_points = find_special_points_of_antiderivative(
            expr, integrals[1], x, x_min, x_max
        )
        area_value = definite_integral(expr, x, *area_bounds) if area_bounds else None
        geometric_area_value = geometric_area(expr, x, *area_bounds) if area_bounds else None
        elapsed = time.perf_counter() - start

        print_results(
            expr, integrals, critical_points, inflection_points,
            area_bounds, area_value, geometric_area_value, elapsed,
        )

        save_path: Optional[str] = None
        if ask_yes_no("Lưu hình đồ thị vào tệp?", default=False):
            save_path = (
                input(f"{C_GREEN}Đường dẫn tệp (ví dụ: ket_qua.png hoặc ket_qua.pdf): {C_RESET}").strip()
                or None
            )

        plot_results(
            expr, integrals, critical_points, inflection_points,
            x, order, (x_min, x_max), area_bounds, save_path,
        )

        if not ask_yes_no("Phân tích một hàm số khác?", default=True):
            print(f"\n{C_CYAN}Session ended.  Tạm biệt!{C_RESET}\n")
            break


if __name__ == "__main__":
    run_session()
