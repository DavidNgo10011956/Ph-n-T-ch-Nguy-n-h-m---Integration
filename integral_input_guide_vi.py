"""Input guide tables and terminal banner for the integral analyser."""

from __future__ import annotations

import re

WIDTH = 88


def hr(ch: str = "═") -> str:
    return ch * WIDTH


# ---------------------------------------------------------------------------
# Special-function catalogue  (shown when the user wants the advanced guide)
# ---------------------------------------------------------------------------

SPECIAL_FUNCTIONS: list[dict[str, str]] = [
    {
        "name":    "erf(x)",
        "symbol":  "erf(x)",
        "meaning": "Hàm lỗi — đo diện tích dưới đường cong hình chuông từ 0 đến x.",
        "used_in": "Thống kê, xác suất và bảng phân phối chuẩn.",
        "example": "∫e⁻ˣ²dx  =  (√π/2)·erf(x) + C",
    },
    {
        "name":    "erfc(x)",
        "symbol":  "erfc(x)",
        "meaning": "Hàm lỗi bù — được định nghĩa là 1 − erf(x).",
        "used_in": "Xác suất phần đuôi và phương trình truyền nhiệt.",
        "example": "erfc(x)  →  xuất hiện khi tích phân e⁻ˣ² trên [x, ∞)",
    },
    {
        "name":    "gamma(x)",
        "symbol":  "Γ(x)",
        "meaning": "Hàm Gamma — mở rộng khái niệm giai thừa cho số không nguyên. Γ(n) = (n−1)!",
        "used_in": "Nâng cao calculus, combinatorics, physics.",
        "example": "gamma(x)  e.g. Γ(5) = 24",
    },
    {
        "name":    "zeta(x)",
        "symbol":  "ζ(x)",
        "meaning": "Hàm Zeta Riemann — tổng 1/nˢ trên các số nguyên dương.",
        "used_in": "Lý thuyết số và vật lý trường lượng tử.",
        "example": "zeta(2)  →  π²/6  ≈ 1.6449",
    },
    {
        "name":    "DiracDelta(x)",
        "symbol":  "δ(x)",
        "meaning": "Delta Dirac — bằng 0 ở mọi nơi trừ x = 0, nơi nó được mô hình hóa như một xung vô hạn.",
        "used_in": "Điện tử, cơ học lượng tử và xử lý tín hiệu.",
        "example": "DiracDelta(x)  →  models an instantaneous impulse",
    },
    {
        "name":    "Heaviside(x)",
        "symbol":  "H(x)",
        "meaning": "Hàm bước Heaviside — bằng 0 khi x < 0 và bằng 1 khi x ≥ 0.",
        "used_in": "Hệ thống điều khiển và phân tích mạch điện.",
        "example": "Heaviside(x)  →  switches from off to on at x = 0",
    },
    {
        "name":    "sinc(x)",
        "symbol":  "sinc(x)",
        "meaning": "Hàm sinc — được định nghĩa là sin(x)/x; theo quy ước sinc(0) = 1.",
        "used_in": "Xử lý tín hiệu, phân tích Fourier và truyền thông.",
        "example": "sinc(x)  →  the shape of a single radio pulse",
    },
]


def special_function_guide() -> str:
    """Return a formatted guide for special / advanced functions."""
    lines: list[str] = [
        "",
        "CÁC HÀM ĐẶC BIỆT  (advanced — most students won't need these)",
        "-" * 72,
        "Các hàm này đã được tích hợp nhưng có thể còn xa lạ với nhiều học sinh.",
        "Bạn vẫn có thể nhập các hàm này — chương trình sẽ tính đạo hàm chính xác",
        "và giải thích ý nghĩa của chúng trong bảng đồ thị.",
        "",
    ]
    for fn in SPECIAL_FUNCTIONS:
        lines.append(f"  {fn['name']:<18}  {fn['symbol']}")
        lines.append(f"  {'':18}  Ý nghĩa    : {fn['meaning']}")
        lines.append(f"  {'':18}  Ứng dụng   : {fn['used_in']}")
        lines.append(f"  {'':18}  Ví dụ      : {fn['example']}")
        lines.append("")
    return "\n".join(lines)


def is_special_function(expr_str: str) -> str | None:
    """Return a plain-English note if a special function is detected in expr_str."""
    checks = {
        "erf":        "erf(x) is the Error Function — related to the normal distribution curve.",
        "erfc":       "erfc(x) is the Complementary Error Function: erfc(x) = 1 − erf(x).",
        "erfi":       "erfi(x) is the Imaginary Error Function: erfi(x) = −i·erf(ix).",
        "gamma":      "Γ(x) is the Gamma Function — a generalisation of the factorial.",
        "zeta":       "ζ(x) is the Riemann Zeta Function — important in number theory. Note: it has no elementary closed-form derivative.",
        "DiracDelta": "δ(x) is the Dirac Delta — an infinitely thin spike at x = 0.",
        "Heaviside":  "H(x) is the Heaviside Step — 0 for x < 0, 1 for x ≥ 0.",
        "sinc":       "sinc(x) = sin(x)/x — a key function in signal processing.",
    }
    # Check longest keys first and match as an actual function call, e.g.
    # "erfc(" — otherwise a plain substring check lets "erf" match inside
    # "erfc(x)" or "erfi(x)" and return the wrong explanation.
    for key in sorted(checks, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\(", expr_str):
            return checks[key]
    return None


# ---------------------------------------------------------------------------
# Standard input / symbol guides
# ---------------------------------------------------------------------------

def input_guide_rows() -> list[dict[str, str]]:
    """Professional guide table to reduce input errors."""
    return [
        {"Biểu thức toán học": "x²",           "Ý nghĩa": "biến x bình phương",             "Cách nhập": "x**2"},
        {"Biểu thức toán học": "x³",           "Ý nghĩa": "biến x lập phương",               "Cách nhập": "x**3"},
        {"Biểu thức toán học": "3x",           "Ý nghĩa": "3 nhân với x",              "Cách nhập": "3*x"},
        {"Biểu thức toán học": "3x²",          "Ý nghĩa": "3 nhân với x squared",      "Cách nhập": "3*x**2"},
        {"Biểu thức toán học": "x² + 3x − 2", "Ý nghĩa": "đa thức",                     "Cách nhập": "x**2 + 3*x - 2"},
        {"Biểu thức toán học": "sin(x)",       "Ý nghĩa": "hàm sin",                  "Cách nhập": "sin(x)"},
        {"Biểu thức toán học": "cos(x)",       "Ý nghĩa": "cohàm sin",                "Cách nhập": "cos(x)"},
        {"Biểu thức toán học": "x sin(x)",     "Ý nghĩa": "x nhân với sin(x)",         "Cách nhập": "x*sin(x)"},
        {"Biểu thức toán học": "x² sin(x)",    "Ý nghĩa": "x² nhân với sin(x)",        "Cách nhập": "x**2*sin(x)"},
        {"Biểu thức toán học": "eˣ",           "Ý nghĩa": "hàm mũ",           "Cách nhập": "exp(x)"},
        {"Biểu thức toán học": "e²ˣ",          "Ý nghĩa": "e mũ 2x",             "Cách nhập": "exp(2*x)"},
        {"Biểu thức toán học": "ln(x)",        "Ý nghĩa": "logarit tự nhiên",              "Cách nhập": "log(x)"},
        {"Biểu thức toán học": "log₂(x)",      "Ý nghĩa": "logarit cơ số 2",                    "Cách nhập": "log(x, 2)"},
        {"Biểu thức toán học": "√x",           "Ý nghĩa": "căn bậc hai",                   "Cách nhập": "sqrt(x)"},
        {"Biểu thức toán học": "x² / (x + 1)","Ý nghĩa": "phân số / thương",            "Cách nhập": "(x**2)/(x + 1)"},
        {"Biểu thức toán học": "sin(x²)",      "Ý nghĩa": "ví dụ quy tắc chuỗi",            "Cách nhập": "sin(x**2)"},
        {"Biểu thức toán học": "|x|",          "Ý nghĩa": "giá trị tuyệt đối",                "Cách nhập": "Abs(x)"},
        {"Biểu thức toán học": "arcsin(x)",    "Ý nghĩa": "hàm sin ngược",                  "Cách nhập": "asin(x)"},
        {"Biểu thức toán học": "arccos(x)",    "Ý nghĩa": "hàm cos ngược",                "Cách nhập": "acos(x)"},
        {"Biểu thức toán học": "arctan(x)",    "Ý nghĩa": "hàm tan ngược",               "Cách nhập": "atan(x)"},
        {"Biểu thức toán học": "sinh(x)",      "Ý nghĩa": "hàm sin hyperbolic",               "Cách nhập": "sinh(x)"},
        {"Biểu thức toán học": "cosh(x)",      "Ý nghĩa": "hàm cos hyperbolic",             "Cách nhập": "cosh(x)"},
    ]


def symbol_guide_rows() -> list[dict[str, str]]:
    """Guide table explaining symbols used by Python/SymPy notation."""
    return [
        {"Ký hiệu": "x",  "Ý nghĩa": "the variable x",         "Ví dụ": "x**2 means x²"},
        {"Ký hiệu": "*",  "Ý nghĩa": "multiplication",          "Ví dụ": "3*x means 3 × x"},
        {"Ký hiệu": "**", "Ý nghĩa": "power / exponent",        "Ví dụ": "x**3 means x³"},
        {"Ký hiệu": "()", "Ý nghĩa": "brackets for grouping",   "Ví dụ": "sin(x), (x+1)*(x-2)"},
        {"Ký hiệu": "/",  "Ý nghĩa": "division",                "Ví dụ": "(x**2)/(x+1)"},
        {"Ký hiệu": "-",  "Ý nghĩa": "minus / subtraction",     "Ví dụ": "x**2 - 3*x + 1"},
    ]


def example_functions() -> list[dict[str, str]]:
    """A curated list of try-me examples, grouped by difficulty."""
    return [
        # Cơ bản
        {"Mức độ": "Cơ bản", "Cách nhập": "x**2",              "Đọc là": "x²"},
        {"Mức độ": "Cơ bản", "Cách nhập": "x**3 - 3*x",       "Đọc là": "x³ − 3x"},
        {"Mức độ": "Cơ bản", "Cách nhập": "2*x**2 + 5*x - 3", "Đọc là": "2x² + 5x − 3"},
        # Trung bình
        {"Mức độ": "Trung bình", "Cách nhập": "sin(x)",          "Đọc là": "sin(x)"},
        {"Mức độ": "Trung bình", "Cách nhập": "exp(x)",           "Đọc là": "eˣ"},
        {"Mức độ": "Trung bình", "Cách nhập": "log(x)",           "Đọc là": "ln(x)"},
        {"Mức độ": "Trung bình", "Cách nhập": "x*sin(x)",         "Đọc là": "x·sin(x)  [product rule]"},
        {"Mức độ": "Trung bình", "Cách nhập": "(x**2)/(x+1)",     "Đọc là": "x²/(x+1)  [quotient rule]"},
        {"Mức độ": "Trung bình", "Cách nhập": "sin(x**2)",        "Đọc là": "sin(x²)   [chain rule]"},
        # Nâng cao
        {"Mức độ": "Nâng cao", "Cách nhập": "exp(-x**2)",          "Đọc là": "e⁻ˣ²  (bell curve shape)"},
        {"Mức độ": "Nâng cao", "Cách nhập": "x**2 * exp(-x)",      "Đọc là": "x²eˣ"},
        {"Mức độ": "Nâng cao", "Cách nhập": "asin(x)",             "Đọc là": "arcsin(x)"},
        {"Mức độ": "Nâng cao", "Cách nhập": "sinh(x)",             "Đọc là": "sinh(x)"},
        # Special
        {"Mức độ": "Special",  "Cách nhập": "erf(x)",              "Đọc là": "Error Function"},
        {"Mức độ": "Special",  "Cách nhập": "Heaviside(x)",        "Đọc là": "Step Function"},
    ]


def _table(headers: list[str], rows: list[dict[str, str]], keys: list[str], widths: list[int]) -> list[str]:
    lines = []
    lines.append("  " + "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    lines.append("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        lines.append("  " + "  ".join(f"{row[k]:<{w}}" for k, w in zip(keys, widths)))
    return lines


def banner(show_special: bool = False) -> str:
    """Return the professional startup/help banner.

    Parameters
    ----------
    show_special:
        When True, append the special-functions reference section.
    """
    lines: list[str] = [
        hr(),
        f"{'PROFESSIONAL INTEGRAL ANALYSER':^{WIDTH}}",
        f"{'Interactive Terminal Edition':^{WIDTH}}",
        hr(),
        "",
        "HƯỚNG DẪN NHẬP",
        "",
    ]
    lines.extend(
        _table(
            ["Biểu thức toán học", "Ý nghĩa", "Cách nhập"],
            input_guide_rows(),
            ["Biểu thức toán học", "Ý nghĩa", "Cách nhập"],
            [16, 38, 24],
        )
    )
    lines.extend(["", "SYMBOL GUIDE", ""])
    lines.extend(
        _table(
            ["Ký hiệu", "Ý nghĩa", "Ví dụ"],
            symbol_guide_rows(),
            ["Ký hiệu", "Ý nghĩa", "Ví dụ"],
            [10, 28, 36],
        )
    )
    lines.extend(["", "CÁC HÀM MẪU ĐỂ THỬ", ""])
    lines.extend(
        _table(
            ["Mức độ", "Cách nhập", "Đọc là"],
            example_functions(),
            ["Mức độ", "Cách nhập", "Đọc là"],
            [14, 28, 36],
        )
    )
    lines.extend(
        [
            "",
            "Commands:",
            "  help           - show this guide again",
            "  special        - show the special / advanced functions guide",
            "  quit  / exit   - close the program",
            "",
            "Tip: x is always the variable. Use * for multiplication, ** for powers.",
            "Tip: The app accepts many natural forms — try  3x^2  or  sin x^2.",
        ]
    )
    if show_special:
        lines.append(special_function_guide())
    lines.append(hr())
    return "\n".join(lines)
