"""Streamlit web front-end for the Professional Integral Analyser.

This is the entry point for Streamlit Community Cloud. It replaces the
terminal input()/print() loop in integral_app_vi.py with Streamlit widgets,
and reuses the existing computation modules unchanged:

    integral_engine_vi.py          — parsing, integration, definite integrals
    integral_point_analysis_vi.py  — critical / inflection point finding
    integral_plotting_vi.py        — builds the Matplotlib figure
    integral_input_guide_vi.py     — help text / example tables
    utils_vi.py                    — human_math() formatting, safe eval

Run locally with:   streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st
from sympy import symbols

from integral_engine_vi import (
    area_working_lines,
    compute_integrals,
    constant_suffix,
    definite_integral,
    first_integral_working,
    geometric_area,
    integral_label,
    integral_title,
    parse_function,
    readable_integral,
    special_function_note,
)
from integral_input_guide_vi import (
    example_functions,
    input_guide_rows,
    is_special_function,
    special_function_guide,
    symbol_guide_rows,
)
from integral_plotting_vi import plot_results
from integral_point_analysis_vi import find_special_points_of_antiderivative
from utils_vi import human_math

X = symbols("x")

st.set_page_config(
    page_title="Professional Integral Analyser",
    page_icon="∫",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Plain-text report (mirrors what the terminal/CLI version prints, plus the
# step-by-step derivation, so results and explanations can be saved/shared
# outside the browser).
# ---------------------------------------------------------------------------

def build_text_report(result: dict) -> str:
    expr = result["expr"]
    integrals = result["integrals"]
    rule = "=" * 88
    sep = "-" * 88
    lines: list[str] = [rule, "PROFESSIONAL INTEGRAL ANALYSER — KẾT QUẢ PHÂN TÍCH", rule, ""]

    lines += ["HÀM SỐ", sep, f"f(x) = {human_math(expr)}"]
    note = special_function_note(str(expr))
    if note:
        lines += ["", f"Ghi chú: {note}"]
    special_note = is_special_function(result["raw_input"])
    if special_note:
        lines += ["", f"Ghi chú: {special_note}"]
    lines.append("")

    lines += ["CÁC BƯỚC TÍNH NGUYÊN HÀM (BẬC 1)", sep]
    lines += first_integral_working(expr, integrals[1], X)
    lines.append("")

    for n, integral_expr in integrals.items():
        disp = readable_integral(integral_expr, n)
        lines += [integral_title(n), sep,
                  f"{integral_label(n)} = {human_math(disp)}   {constant_suffix(n)}", ""]

    if result["area_bounds"]:
        a_, b_ = result["area_bounds"]
        lines += ["TÍCH PHÂN XÁC ĐỊNH VÀ DIỆN TÍCH", sep]
        if result["area_value"] is None:
            lines.append(f"Không thể tính tích phân xác định trên [{a_}, {b_}].")
        else:
            lines.append(f"DIỆN TÍCH CÓ DẤU    ∫[{a_}, {b_}] f(x) dx   = {result['area_value']:.6f}")
        if result["geom_value"] is None:
            lines.append(f"Không thể tính diện tích hình học trên [{a_}, {b_}].")
        else:
            lines.append(f"DIỆN TÍCH HÌNH HỌC  ∫[{a_}, {b_}] |f(x)| dx = {result['geom_value']:.6f}")
        lines += ["", "Các bước tính chi tiết:"]
        lines += area_working_lines(expr, X, a_, b_)
        lines.append("")

    lines += ["CÁC ĐIỂM ĐẶC BIỆT CỦA F(x)", sep]
    critical_points = result["critical_points"]
    inflection_points = result["inflection_points"]
    total = len(critical_points) + len(inflection_points)
    lines.append(f"Tổng số điểm đặc biệt khác nhau tìm được: {total}")
    lines.append("(Cực trị của F tại nơi f(x) = 0 đổi dấu; điểm uốn của F tại nơi f′(x) = 0 đổi dấu)")
    if critical_points:
        for xv, yv, kind in critical_points:
            lines.append(f"  {kind:<24} x = {xv: .6f},   F(x) = {yv: .6f}")
    else:
        lines.append("  Không tìm thấy cực trị của F(x) trong khoảng x này.")
    if inflection_points:
        for xv, yv in inflection_points:
            lines.append(f"  {'Điểm uốn':<24} x = {xv: .6f},   F(x) = {yv: .6f}")
    else:
        lines.append("  Không tìm thấy điểm uốn của F(x) trong khoảng x này.")
    lines.append("")

    lines += ["THÔNG SỐ ĐẦU VÀO", sep,
              f"f(x) nhập vào       : {result['raw_input']}",
              f"Bậc nguyên hàm       : {result['order']}",
              f"Khoảng x của đồ thị  : [{result['x_min']}, {result['x_max']}]", ""]

    lines += ["THỜI GIAN XỬ LÝ", sep, f"{result['elapsed']:.4f} seconds", rule]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar — input guide (reuses the same tables the CLI banner shows)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Hướng dẫn nhập")
    st.dataframe(pd.DataFrame(input_guide_rows()), hide_index=True, width="stretch")

    st.header("Ký hiệu")
    st.dataframe(pd.DataFrame(symbol_guide_rows()), hide_index=True, width="stretch")

    st.header("Ví dụ để thử")
    st.dataframe(pd.DataFrame(example_functions()), hide_index=True, width="stretch")

    with st.expander("Các hàm đặc biệt (nâng cao)"):
        st.text(special_function_guide())


# ---------------------------------------------------------------------------
# Main input form
# ---------------------------------------------------------------------------

st.title("∫ Professional Integral Analyser")
st.caption("Nhập f(x) để tính nguyên hàm, tích phân xác định và các điểm đặc biệt của F(x).")

with st.form("integral_form"):
    function_text = st.text_input(
        "Nhập f(x):",
        value="x**2 - 3*x + 1",
        help="Ví dụ: sin(x**2), x*exp(-x), (x**2)/(x+1), erf(x)",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        order = st.selectbox(
            "Bậc nguyên hàm",
            [1, 2, 3],
            index=0,
            help="1 = ∫f dx,  2 = ∫∫f dx dx,  3 = ∫∫∫f dx dx dx",
        )
    with col2:
        x_min = st.number_input("x_min (đồ thị)", value=-6.0, step=1.0)
    with col3:
        x_max = st.number_input("x_max (đồ thị)", value=6.0, step=1.0)

    want_area = st.checkbox("Tính tích phân xác định (diện tích) trên một khoảng?")
    a_bound = b_bound = None
    if want_area:
        col4, col5 = st.columns(2)
        with col4:
            a_bound = st.number_input("Cận dưới a", value=float(x_min))
        with col5:
            b_bound = st.number_input("Cận trên b", value=float(x_max))

    submitted = st.form_submit_button("Tính toán", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Validation + computation
# ---------------------------------------------------------------------------

if submitted:
    if x_min >= x_max:
        st.error("x_min phải nhỏ hơn x_max.")
    elif want_area and (a_bound is None or b_bound is None or a_bound >= b_bound):
        st.error("Cận dưới a phải nhỏ hơn cận trên b.")
    else:
        # parse_function() reports errors via print() rather than raising —
        # capture stdout so those messages can be shown as a proper st.error
        # instead of vanishing into the server's console log.
        capture = io.StringIO()
        with redirect_stdout(capture):
            expr = parse_function(function_text, X)

        if expr is None:
            st.error("Không thể phân tích biểu thức này.")
            details = capture.getvalue().strip()
            if details:
                st.code(details)
            st.info(
                "Ví dụ hợp lệ: x**3 - 2*x + 1,  sin(x**2),  x*sin(x),  exp(-x**2). "
                "Xem thêm hướng dẫn ở thanh bên trái."
            )
            st.session_state.pop("result", None)
        else:
            area_bounds = (a_bound, b_bound) if want_area else None
            with st.spinner("Đang tính toán…"):
                start = time.perf_counter()
                integrals = compute_integrals(expr, X, order)
                critical_points, inflection_points = find_special_points_of_antiderivative(
                    expr, integrals[1], X, x_min, x_max
                )
                area_value = definite_integral(expr, X, *area_bounds) if area_bounds else None
                geom_value = geometric_area(expr, X, *area_bounds) if area_bounds else None
                elapsed = time.perf_counter() - start

                fig = plot_results(
                    expr, integrals, critical_points, inflection_points,
                    X, order, (x_min, x_max), area_bounds, save_path=None,
                )

            st.session_state["result"] = {
                "expr": expr,
                "integrals": integrals,
                "critical_points": critical_points,
                "inflection_points": inflection_points,
                "area_bounds": area_bounds,
                "area_value": area_value,
                "geom_value": geom_value,
                "elapsed": elapsed,
                "fig": fig,
                "raw_input": function_text,
                "order": order,
                "x_min": x_min,
                "x_max": x_max,
            }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.get("result")

if result is None:
    st.info("Nhập một hàm số f(x) ở trên và nhấn **Tính toán** để bắt đầu.")
else:
    expr = result["expr"]
    integrals = result["integrals"]

    st.divider()

    st.subheader("Hàm số")
    st.markdown(f"### f(x) = {human_math(expr)}")

    note = special_function_note(str(expr))
    if note:
        st.info(note)

    special_note = is_special_function(result["raw_input"])
    if special_note:
        st.info(special_note)

    st.subheader("Nguyên hàm")
    with st.expander("📝 Các bước tính chi tiết (bậc 1)", expanded=False):
        for line in first_integral_working(expr, integrals[1], X):
            st.text(line)

    for n, integral_expr in integrals.items():
        display_integral = readable_integral(integral_expr, n)
        st.markdown(f"**{integral_title(n)}**")
        st.code(
            f"{integral_label(n)} = {human_math(display_integral)}   {constant_suffix(n)}",
            language=None,
        )

    if result["area_bounds"]:
        a_, b_ = result["area_bounds"]
        st.subheader("Tích phân xác định & diện tích")
        col1, col2 = st.columns(2)
        with col1:
            if result["area_value"] is None:
                st.warning(f"Không thể tính tích phân xác định trên [{a_}, {b_}].")
            else:
                st.metric(f"Diện tích có dấu  ∫[{a_:g}, {b_:g}] f(x) dx", f"{result['area_value']:.6f}")
        with col2:
            if result["geom_value"] is None:
                st.warning(f"Không thể tính diện tích hình học trên [{a_}, {b_}].")
            else:
                st.metric(f"Diện tích hình học  ∫[{a_:g}, {b_:g}] |f(x)| dx", f"{result['geom_value']:.6f}")

        with st.expander("Các bước tính chi tiết"):
            for line in area_working_lines(expr, X, a_, b_):
                st.text(line)

    st.subheader("Các điểm đặc biệt của F(x)")
    critical_points = result["critical_points"]
    inflection_points = result["inflection_points"]
    total = len(critical_points) + len(inflection_points)
    st.write(f"Tổng số điểm đặc biệt khác nhau tìm được: **{total}**")
    st.caption("Cực trị của F tại nơi f(x) = 0 đổi dấu; điểm uốn của F tại nơi f′(x) = 0 đổi dấu.")

    if critical_points or inflection_points:
        rows = [
            {"Loại": kind, "x": f"{xv:.6f}", "F(x)": f"{yv:.6f}"}
            for xv, yv, kind in critical_points
        ] + [
            {"Loại": "Điểm uốn", "x": f"{xv:.6f}", "F(x)": f"{yv:.6f}"}
            for xv, yv in inflection_points
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.write("Không tìm thấy điểm đặc biệt nào trong khoảng x này.")

    st.subheader("Đồ thị")
    st.pyplot(result["fig"])

    png_buffer = io.BytesIO()
    result["fig"].savefig(png_buffer, format="png", dpi=180, bbox_inches="tight")

    report_text = build_text_report(result)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇ Tải hình đồ thị (PNG)",
            data=png_buffer.getvalue(),
            file_name="ket_qua_tich_phan.png",
            mime="image/png",
            width="stretch",
        )
    with dl_col2:
        st.download_button(
            "⬇ Tải kết quả & giải thích (TXT)",
            data=report_text,
            file_name="ket_qua_tich_phan.txt",
            mime="text/plain",
            width="stretch",
        )

    with st.expander("Xem trước nội dung tệp TXT"):
        st.text(report_text)

    st.caption(f"Thời gian xử lý: {result['elapsed']:.4f} giây")