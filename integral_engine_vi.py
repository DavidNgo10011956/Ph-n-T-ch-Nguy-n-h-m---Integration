"""Phân tích biểu thức, tính tích phân (nguyên hàm) và trình bày các bước tính toán.

Đây là phiên bản "đảo ngược" của derivative_engine_vi.py — thay vì lấy đạo
hàm, mô-đun này lấy TÍCH PHÂN (nguyên hàm bất định, và tích phân xác định
khi người dùng cung cấp cận).
"""

from __future__ import annotations

import re
from typing import Optional

from sympy import (
    E,
    Expr,
    Integer,
    Integral,
    Mul,
    Rational,
    Symbol,
    cos,
    diff,
    exp,
    expand,
    integrate,
    log,
    nan,
    oo,
    pi,
    simplify,
    sin,
    sqrt,
    sympify,
    tan,
    zoo,
    sinh,
    cosh,
    tanh,
    asin,
    acos,
    atan,
    Abs,
    singularities,
    limit,
    together,
    fraction,
)

from utils_vi import C_RED, C_RESET, C_YELLOW, human_math, to_superscript

SYMPY_LOCALS: dict[str, object] = {
    "sin": sin,   "cos": cos,   "tan": tan,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "asin": asin, "acos": acos, "atan": atan,
    "exp": exp,   "log": log,   "sqrt": sqrt,
    "Abs": Abs,
    "pi": pi,     "E": E,
}


def _load_specials() -> dict[str, object]:
    specials: dict[str, object] = {}
    try:
        from sympy import erf, erfc, erfi, gamma, zeta, LambertW
        from sympy.functions.special.gamma_functions import digamma
        specials.update({
            "erf": erf, "erfc": erfc, "erfi": erfi,
            "gamma": gamma, "zeta": zeta, "LambertW": LambertW,
            "digamma": digamma,
        })
    except ImportError:
        pass
    try:
        from sympy.functions.special.delta_functions import Heaviside, DiracDelta
        specials.update({"Heaviside": Heaviside, "DiracDelta": DiracDelta})
    except ImportError:
        pass
    try:
        from sympy import sinc
        specials["sinc"] = sinc
    except ImportError:
        pass
    return specials


SYMPY_LOCALS.update(_load_specials())


# ---------------------------------------------------------------------------
# Ghi chú tiếng Việt khi một hàm đặc biệt xuất hiện trong kết quả
# ---------------------------------------------------------------------------

SPECIAL_FUNCTION_NOTES: dict[str, str] = {
    "erf":        "Ghi chú: erf là Hàm lỗi (Error Function). ∫e⁻ˣ²dx không có dạng sơ cấp — kết quả được biểu diễn qua erf(x): ∫e⁻ˣ²dx = (√π/2)·erf(x) + C.",
    "erfc":       "Ghi chú: erfc là Hàm lỗi bù (= 1 − erf(x)); thường xuất hiện khi tích phân e⁻ˣ² trên khoảng vô hạn.",
    "erfi":       "Ghi chú: erfi là Hàm lỗi ảo; xuất hiện khi tích phân eˣ².",
    "gamma":      "Ghi chú: Γ là Hàm Gamma. ∫₀^∞ tˣ⁻¹e⁻ᵗdt = Γ(x) — một trong những tích phân suy rộng nổi tiếng nhất.",
    "zeta":       "Ghi chú: ζ là Hàm Zeta Riemann; không có nguyên hàm sơ cấp.",
    "DiracDelta": "Ghi chú: δ(x) là Delta Dirac. ∫δ(x)dx = H(x), hàm bước Heaviside.",
    "Heaviside":  "Ghi chú: H(x) là hàm bước Heaviside. ∫H(x)dx = x·H(x) (bằng 0 khi x<0, bằng x khi x≥0).",
    "sinc":       "Ghi chú: sinc(x) = sin(x)/x; nguyên hàm liên quan đến hàm Si(x) (Sine Integral), không sơ cấp.",
    "LambertW":   "Ghi chú: W(x) là Hàm Lambert W.",
}

_SPECIAL_NOTE_KEYS_BY_LENGTH: list[str] = sorted(SPECIAL_FUNCTION_NOTES, key=len, reverse=True)


def special_function_note(expr_str: str) -> str | None:
    """Trả về ghi chú tiếng Việt đầu tiên nếu có hàm đặc biệt xuất hiện."""
    for key in _SPECIAL_NOTE_KEYS_BY_LENGTH:
        if re.search(rf"\b{re.escape(key)}\(", expr_str):
            return SPECIAL_FUNCTION_NOTES[key]
    return None


def special_function_notes_for_expr(expr: Expr) -> list[str]:
    """Trả về ghi chú cho MỌI hàm đặc biệt xuất hiện trong biểu thức (dùng trong bảng bước)."""
    text = str(expr)
    notes: list[str] = []
    for key in _SPECIAL_NOTE_KEYS_BY_LENGTH:
        if re.search(rf"\b{re.escape(key)}\(", text):
            notes.append(SPECIAL_FUNCTION_NOTES[key])
    return notes


# ---------------------------------------------------------------------------
# Chuẩn hóa dữ liệu nhập (giống hệt bản đạo hàm)
# ---------------------------------------------------------------------------

def normalise_input(raw: str) -> str:
    """Chuyển ký hiệu toán học thông dụng thành cú pháp hợp lệ của Python/SymPy."""
    text = raw.strip()
    text = text.replace("×", "*")
    text = text.replace("^", "**")
    text = re.sub(r"\bX\b", "x", text)

    replacements = {
        "Sin": "sin", "Cos": "cos", "Tan": "tan", "Exp": "exp",
        "Log": "log", "Ln": "log", "Sqrt": "sqrt",
        "SIN": "sin", "COS": "cos", "TAN": "tan", "EXP": "exp",
        "LOG": "log", "LN": "log", "SQRT": "sqrt",
        "Sinh": "sinh", "Cosh": "cosh", "Tanh": "tanh",
        "SINH": "sinh", "COSH": "cosh", "TANH": "tanh",
        "Asin": "asin", "Acos": "acos", "Atan": "atan",
        "ASIN": "asin", "ACOS": "acos", "ATAN": "atan",
        "arcsin": "asin", "arccos": "acos", "arctan": "atan",
        "ArcSin": "asin", "ArcCos": "acos", "ArcTan": "atan",
        "Erf": "erf", "Erfc": "erfc", "Erfi": "erfi",
        "Gamma": "gamma", "Zeta": "zeta", "Sinc": "sinc",
        "H": "Heaviside", "Step": "Heaviside",
        "Delta": "DiracDelta", "Diracdelta": "DiracDelta",
        "lambertw": "LambertW", "Lambertw": "LambertW",
        "Abs": "Abs", "abs": "Abs",
    }
    for old, new in replacements.items():
        text = re.sub(rf"\b{old}\b", new, text)

    text = re.sub(r"(\d)(x)", r"\1*\2", text)
    text = re.sub(r"(\d)\(", r"\1*(", text)
    text = re.sub(r"(x)\(", r"\1*(", text)
    known_function_names = "|".join(sorted(map(re.escape, SYMPY_LOCALS), key=len, reverse=True))
    text = re.sub(rf"(\d)({known_function_names})", r"\1*\2", text)
    text = re.sub(r"(?<![A-Za-z0-9_])x(\d)", r"x*\1", text)

    return text


# ---------------------------------------------------------------------------
# Phân tích biểu thức
# ---------------------------------------------------------------------------

def parse_function(raw: str, x: Symbol) -> Optional[Expr]:
    """Phân tích dữ liệu nhập thành biểu thức SymPy hợp lệ chỉ chứa biến x."""
    try:
        expr = sympify(normalise_input(raw), locals={"x": x, **SYMPY_LOCALS})
        if expr in (oo, -oo, zoo, nan):
            print(f"{C_RED}Lỗi nhập liệu: expression is undefined.{C_RESET}")
            return None
        unknown_symbols = expr.free_symbols - {x}
        if unknown_symbols:
            names = ", ".join(sorted(str(s) for s in unknown_symbols))
            print(f"{C_RED}Lỗi nhập liệu: unknown variable(s): {names}.  Use only 'x'.{C_RESET}")
            looks_like_x_plus_digits = any(re.match(r"^x\d+$", str(s)) for s in unknown_symbols)
            if looks_like_x_plus_digits:
                print(f"{C_YELLOW}Tip: '{names}' looks like the variable x written right next to a "
                      f"number. If you meant x multiplied by that number, add a * — "
                      f"e.g. write x*2 instead of x2.{C_RESET}")
            else:
                print(f"{C_YELLOW}Tip: check you haven't used a capital letter (e.g. X instead of x), "
                      f"or added a * for multiplication (e.g. 2*x instead of 2x).{C_RESET}")
            return None
        # Quick validation — attempt a symbolic integral (may stay unevaluated
        # for genuinely non-elementary functions like zeta, which is fine).
        integrate(expr, x)
        return expr
    except Exception as err:
        print(f"{C_RED}Lỗi nhập liệu: {err}{C_RESET}")
        print(f"{C_YELLOW}Tip: type 'help' to see input examples.{C_RESET}")
        return None


# ---------------------------------------------------------------------------
# Tính các nguyên hàm (tích phân bất định lặp) + tích phân xác định
# ---------------------------------------------------------------------------

def compute_integrals(expr: Expr, x: Symbol, order: int) -> dict[int, Expr]:
    """Tính nguyên hàm lặp từ bậc 1 đến bậc được yêu cầu.

    order=1  → F(x)  = ∫ f(x) dx
    order=2  → F2(x) = ∫ F(x) dx   (nguyên hàm của nguyên hàm)
    order=3  → F3(x) = ∫ F2(x) dx
    (Hằng số tích phân được cộng riêng trong phần hiển thị — xem
    `constant_suffix`.)
    """
    integrals: dict[int, Expr] = {}
    current = expr
    for n in range(1, order + 1):
        current = simplify(integrate(current, x))
        integrals[n] = current
    return integrals


def constant_suffix(order: int) -> str:
    """Trả về hậu tố hằng số tích phân phù hợp cho bậc lặp thứ `order`.

    Bậc 1: '+ C'
    Bậc 2: '+ C₁x + C₂'
    Bậc 3: '+ C₁x²/2 + C₂x + C₃'   (dạng tổng quát của nghiệm thuần nhất)
    """
    if order <= 1:
        return "+ C"
    # Build C1*x^(order-1)/(order-1)! + ... + C_order
    from math import factorial
    parts = []
    for i in range(order - 1, 0, -1):
        idx = order - i
        coeff = f"C{to_subscript_idx(idx)}"
        if i == 1:
            parts.append(f"{coeff}x")
        else:
            denom = factorial(i)
            denom_str = "" if denom == 1 else f"/{denom}"
            power_str = to_superscript(str(i))
            parts.append(f"{coeff}x{power_str}{denom_str}")
    parts.append(f"C{to_subscript_idx(order)}")
    return "+ " + " + ".join(parts)


def to_subscript_idx(n: int) -> str:
    from utils_vi import to_subscript
    return to_subscript(n)


def _finite_real_float(value) -> Optional[float]:
    """Đổi một kết quả SymPy thành số thực hữu hạn; trả về None nếu phân kỳ."""
    try:
        numeric = complex(value.evalf())
        if abs(numeric.imag) > 1e-7:
            return None
        real = float(numeric.real)
        import math
        return real if math.isfinite(real) else None
    except Exception:
        return None


def singular_points_in_interval(expr: Expr, x: Symbol, a: float, b: float) -> list[Expr]:
    """Tìm các điểm mà f(x) không xác định trong đoạn đóng [a, b]."""
    low, high = sorted((float(a), float(b)))
    try:
        points = singularities(expr, x)
        found: list[Expr] = []
        for point in points:
            if point.is_real is False or point.is_finite is False:
                continue
            try:
                value = float(point.evalf())
            except Exception:
                continue
            if low - 1e-12 <= value <= high + 1e-12:
                found.append(point)
        return sorted(found, key=lambda p: float(p.evalf()))
    except Exception:
        return []


def _improper_integral_value(expr: Expr, x: Symbol, a: float, b: float) -> Optional[float]:
    """Tính tích phân suy rộng bằng cách tách tại mọi điểm gián đoạn.

    Mỗi phần phải hội tụ riêng. Vì vậy hàm không chấp nhận việc hai vô cực
    trái dấu triệt tiêu theo giá trị chính Cauchy.
    """
    reverse = b < a
    low, high = (b, a) if reverse else (a, b)
    points = singular_points_in_interval(expr, x, low, high)
    cuts = [low, *points, high]
    total = 0.0

    for left, right in zip(cuts, cuts[1:]):
        if left == right:
            continue
        try:
            piece = integrate(expr, (x, left, right))
            value = _finite_real_float(piece)
        except Exception:
            value = None
        if value is None:
            return None
        total += value

    return -total if reverse else total


def definite_integral(expr: Expr, x: Symbol, a: float, b: float) -> Optional[float]:
    """Tính diện tích có dấu ∫ₐᵇf(x)dx, kể cả tích phân suy rộng hội tụ."""
    value = _improper_integral_value(expr, x, a, b)
    if value is not None:
        return value

    # Chỉ dùng tích phân số khi không có điểm kỳ dị trong khoảng.
    if singular_points_in_interval(expr, x, a, b):
        return None
    try:
        from scipy import integrate as scipy_integrate
        from utils_vi import safe_lambdify
        f = safe_lambdify(x, expr)
        value, _ = scipy_integrate.quad(lambda v: float(f(v)), a, b, limit=300)
        import math
        return float(value) if math.isfinite(value) else None
    except Exception:
        return None


def geometric_area(expr: Expr, x: Symbol, a: float, b: float) -> Optional[float]:
    """Tính diện tích hình học ∫ₐᵇ|f(x)|dx, kể cả tích phân suy rộng hội tụ."""
    value = _improper_integral_value(Abs(expr), x, a, b)
    if value is not None:
        return abs(value)

    if singular_points_in_interval(expr, x, a, b):
        return None
    try:
        from scipy import integrate as scipy_integrate
        from utils_vi import safe_lambdify
        f = safe_lambdify(x, expr)
        value, _ = scipy_integrate.quad(lambda v: abs(float(f(v))), a, b, limit=400)
        import math
        return float(abs(value)) if math.isfinite(value) else None
    except Exception:
        return None


def area_education_lines(expr: Expr, x: Symbol, a: float, b: float) -> list[str]:
    """Giải thích ngắn gọn vì sao diện tích tính được hoặc không tính được."""
    signed = definite_integral(expr, x, a, b)
    geometric = geometric_area(expr, x, a, b)
    points = singular_points_in_interval(expr, x, a, b)
    lines: list[str] = []

    if points:
        shown = ", ".join(human_math(p) for p in points)
        lines.append(f"Hàm số không liên tục trong khoảng tại x = {shown}.")

        # Nêu nguyên nhân mẫu số bằng 0 khi có thể nhận diện được.
        try:
            _, denominator = fraction(together(expr))
            zero_points = [p for p in points if simplify(denominator.subs(x, p)) == 0]
            if zero_points:
                lines.append("Tại đó mẫu số bằng 0 nên hàm số không xác định.")
        except Exception:
            pass

        asymptotes: list[str] = []
        for p in points:
            try:
                left_lim = limit(expr, x, p, dir="-")
                right_lim = limit(expr, x, p, dir="+")
                if left_lim in (oo, -oo) or right_lim in (oo, -oo):
                    asymptotes.append(human_math(p))
            except Exception:
                continue
        if asymptotes:
            lines.append(f"Đồ thị có tiệm cận đứng tại x = {', '.join(asymptotes)}.")
        lines.append("Phải tách tích phân tại điểm gián đoạn và xét từng giới hạn một phía.")

    if signed is None:
        lines.append("Diện tích có dấu không tồn tại dưới dạng một số hữu hạn vì tích phân suy rộng phân kỳ.")
    else:
        lines.append("Tích phân có dấu hội tụ và cho một giá trị hữu hạn.")

    if geometric is None:
        lines.append("Diện tích hình học cũng vô hạn hoặc phân kỳ vì ∫|f(x)|dx không hội tụ.")
    else:
        lines.append("Diện tích hình học hội tụ; mọi phần được tính dương bằng ∫|f(x)|dx.")

    if not points and signed is None:
        lines.append("Có thể hàm số tăng quá nhanh, có miền xác định phức tạp, hoặc công cụ số không hội tụ.")
    return lines



def area_working_lines(expr: Expr, x: Symbol, a: float, b: float) -> list[str]:
    """Tạo các bước tính diện tích có dấu và diện tích hình học.

    - Diện tích có dấu dùng định lý cơ bản: F(b) - F(a).
    - Diện tích hình học tách khoảng tại các nghiệm của f(x), rồi cộng
      trị tuyệt đối của tích phân trên từng khoảng.
    - Nếu có điểm kỳ dị/phân kỳ, trả về phần giải thích tích phân suy rộng.
    """
    lines: list[str] = []
    signed = definite_integral(expr, x, a, b)
    geometric = geometric_area(expr, x, a, b)
    singular_points = singular_points_in_interval(expr, x, a, b)

    lines.append("A. CÁC BƯỚC TÍNH DIỆN TÍCH CÓ DẤU")
    lines.append(f"   I = ∫[{a:g}, {b:g}] f(x) dx")

    if singular_points:
        shown = ", ".join(human_math(point) for point in singular_points)
        lines.append(f"   Bước 1: Hàm số gián đoạn tại x = {shown}.")
        lines.append("   Bước 2: Tách tích phân tại mỗi điểm gián đoạn và xét giới hạn một phía.")
        for text in area_education_lines(expr, x, a, b):
            lines.append(f"   • {text}")
        if signed is None:
            lines.append("   Kết luận: diện tích có dấu không tồn tại dưới dạng số hữu hạn.")
        else:
            lines.append(f"   Kết luận: tích phân suy rộng hội tụ, I = {signed:.6f}.")
    else:
        try:
            F = simplify(integrate(expr, x))
            Fa = simplify(F.subs(x, a))
            Fb = simplify(F.subs(x, b))
            lines.append(f"   Bước 1: Tìm nguyên hàm F(x) = {human_math(F)}.")
            lines.append("   Bước 2: Áp dụng định lý cơ bản của giải tích:")
            lines.append(f"            I = F({b:g}) − F({a:g})")
            lines.append(f"              = ({human_math(Fb)}) − ({human_math(Fa)})")
            if signed is not None:
                lines.append(f"              = {signed:.6f}")
                if signed > 1e-9:
                    lines.append("   Kết luận: kết quả dương, phần phía trên trục Ox chiếm ưu thế.")
                elif signed < -1e-9:
                    lines.append("   Kết luận: kết quả âm, phần phía dưới trục Ox chiếm ưu thế.")
                else:
                    lines.append("   Kết luận: phần dương và phần âm triệt tiêu nhau.")
            else:
                lines.append("   Kết luận: không thu được giá trị hữu hạn.")
        except Exception:
            if signed is not None:
                lines.append(f"   Tính trực tiếp bằng công cụ đại số/số: I = {signed:.6f}.")
            else:
                lines.append("   Không thể tính được giá trị hữu hạn.")

    lines.append("")
    lines.append("B. CÁC BƯỚC TÍNH DIỆN TÍCH HÌNH HỌC")
    lines.append(f"   A = ∫[{a:g}, {b:g}] |f(x)| dx")

    if geometric is None:
        if singular_points:
            lines.append("   Vì ∫|f(x)|dx không hội tụ nên diện tích hình học là vô hạn hoặc không tồn tại.")
        else:
            lines.append("   Không thể xác định diện tích hình học hữu hạn trên khoảng đã chọn.")
        return lines

    # Find roots strictly inside the interval so the geometric area can be split.
    roots: list[float] = []
    try:
        from integral_point_analysis_vi import numerical_roots
        roots = [r for r in numerical_roots(expr, x, min(a, b), max(a, b))
                 if min(a, b) + 1e-8 < r < max(a, b) - 1e-8]
    except Exception:
        roots = []

    points = [float(a)] + sorted(roots) + [float(b)]
    if roots:
        roots_text = ", ".join(f"{r:.6g}" for r in roots)
        lines.append(f"   Bước 1: Tìm các giao điểm với trục Ox trong khoảng: x = {roots_text}.")
        lines.append("   Bước 2: Chia khoảng tại các giao điểm và xác định dấu của f(x).")
        total = 0.0
        for idx, (left, right) in enumerate(zip(points[:-1], points[1:]), start=1):
            mid = (left + right) / 2
            try:
                sign_value = float(expr.subs(x, mid).evalf())
            except Exception:
                sign_value = 0.0
            piece = definite_integral(expr, x, left, right)
            if piece is None:
                lines.append(f"   Khoảng {idx}: [{left:.6g}, {right:.6g}] không tính được.")
                continue
            area_piece = abs(piece)
            total += area_piece
            sign_text = "dương" if sign_value >= 0 else "âm"
            operation = "giữ nguyên" if piece >= 0 else "đổi dấu"
            lines.append(
                f"   Khoảng {idx}: [{left:.6g}, {right:.6g}], f(x) {sign_text}; "
                f"∫f(x)dx = {piece:.6f} → {operation} → diện tích = {area_piece:.6f}."
            )
        lines.append("   Bước 3: Cộng tất cả các phần diện tích dương:")
        pieces = []
        for left, right in zip(points[:-1], points[1:]):
            value = definite_integral(expr, x, left, right)
            if value is not None:
                pieces.append(f"{abs(value):.6f}")
        if pieces:
            lines.append(f"            A = {' + '.join(pieces)}")
        lines.append(f"              = {geometric:.6f}")
    else:
        lines.append("   Bước 1: f(x) không đổi dấu trong khoảng (không có nghiệm bên trong khoảng).")
        if signed is not None and signed >= 0:
            lines.append("   Bước 2: f(x) ≥ 0 trên khoảng nên |f(x)| = f(x).")
        elif signed is not None:
            lines.append("   Bước 2: f(x) ≤ 0 trên khoảng nên |f(x)| = −f(x).")
        else:
            lines.append("   Bước 2: Lấy trị tuyệt đối của hàm số trên toàn khoảng.")
        lines.append(f"   Bước 3: A = ∫[{a:g}, {b:g}] |f(x)|dx = {geometric:.6f}.")

    lines.append("   Kết luận: diện tích hình học luôn không âm vì mọi phần đều được cộng dương.")
    return lines

def integral_label(order: int) -> str:
    labels = {1: "F(x)", 2: "F₂(x)", 3: "F₃(x)"}
    return labels.get(order, f"F₍{order}₎(x)")


def integral_title(order: int) -> str:
    titles = {
        1: "NGUYÊN HÀM (TÍCH PHÂN BẤT ĐỊNH)",
        2: "NGUYÊN HÀM LẶP BẬC HAI  (∫∫ f dx dx)",
        3: "NGUYÊN HÀM LẶP BẬC BA   (∫∫∫ f dx dx dx)",
    }
    return titles.get(order, f"NGUYÊN HÀM LẶP BẬC {order}")


def readable_integral(expr: Expr, order: int = 1) -> Expr:
    """Khai triển nguyên hàm bậc nhất để học sinh dễ đọc hơn."""
    return expand(expr) if order == 1 else expr


# ---------------------------------------------------------------------------
# Các bước tính nguyên hàm bậc nhất dành cho người mới học
# ---------------------------------------------------------------------------

def first_integral_working(expr: Expr, antiderivative: Expr, x: Symbol) -> list[str]:
    """Trả về các bước tính nguyên hàm rõ ràng, phù hợp người mới học."""
    learning_notes = special_function_notes_for_expr(expr)

    if expr.is_Add:
        lines = _sum_rule_working(expr, antiderivative, x)
    elif expr.is_Pow and expr.base == x and expr.exp.is_number:
        lines = _power_rule_working(expr, antiderivative, x)
    elif _is_simple_table_lookup(expr, x):
        lines = _table_lookup_working(expr, antiderivative, x)
    elif expr.is_Mul:
        u_sub_lines = _substitution_working(expr, antiderivative, x)
        if u_sub_lines:
            lines = u_sub_lines
        else:
            parts_lines = _by_parts_working(expr, antiderivative, x)
            lines = parts_lines if parts_lines else _generic_working(expr, antiderivative)
    elif _is_chain_like(expr, x):
        u_sub_lines = _substitution_working(expr, antiderivative, x)
        lines = u_sub_lines if u_sub_lines else _generic_working(expr, antiderivative)
    else:
        lines = _generic_working(expr, antiderivative)

    if learning_notes:
        lines.extend(["", "Giải thích thêm về các hàm nâng cao / hàm đặc biệt:", ""])
        lines.extend(learning_notes)
    return lines


def _safe_expand(expr: Expr) -> Expr:
    try:
        return expand(expr)
    except Exception:
        return expr


# --- Quy tắc tổng / hiệu --------------------------------------------------

def _sum_rule_working(expr: Expr, antiderivative: Expr, x: Symbol) -> list[str]:
    lines = [
        "Các bước tính nguyên hàm:",
        "Quy tắc sử dụng: QUY TẮC TỔNG / HIỆU",
        "",
        "Khi f(x) = u(x) ± v(x),  thì  ∫f(x)dx = ∫u(x)dx ± ∫v(x)dx",
        "Lấy nguyên hàm từng hạng tử riêng biệt, sau đó cộng hoặc trừ các kết quả.",
        "",
        f"f(x) = {human_math(expr)}",
        "",
    ]
    for term in expr.args:
        F_term = simplify(integrate(term, x))
        lines.append(f"  ∫ {human_math(term)} dx  =  {human_math(_safe_expand(F_term))}")
    lines.extend([
        "",
        f"Kết quả cuối cùng:  ∫f(x)dx = {human_math(_safe_expand(antiderivative))} + C",
    ])
    return lines


# --- Quy tắc lũy thừa -------------------------------------------------------

def _power_rule_working(expr: Expr, antiderivative: Expr, x: Symbol) -> list[str]:
    power = expr.exp
    final_answer = human_math(_safe_expand(antiderivative))

    if power == -1:
        return [
            "Các bước tính nguyên hàm:",
            "Quy tắc sử dụng: TRƯỜNG HỢP ĐẶC BIỆT  n = −1",
            "",
            "Công thức:  ∫x⁻¹dx = ∫(1/x)dx = ln|x| + C",
            "(Quy tắc lũy thừa thông thường không dùng được vì n + 1 = 0.)",
            "",
            f"Kết quả cuối cùng:  ∫f(x)dx = {final_answer} + C",
        ]

    new_power = simplify(power + 1)
    return [
        "Các bước tính nguyên hàm:",
        "Quy tắc sử dụng: QUY TẮC LŨY THỪA (đảo ngược)",
        "",
        "Công thức:  ∫xⁿdx = xⁿ⁺¹/(n + 1) + C   (với n ≠ −1)",
        "Tăng số mũ lên 1, rồi chia biểu thức cho số mũ mới.",
        "",
        f"Bước 1 — Xác định số mũ:  n = {human_math(power)}",
        "",
        f"Bước 2 — Tăng số mũ lên 1:  n + 1 = {human_math(power)} + 1 = {human_math(new_power)}",
        "",
        f"Bước 3 — Chia cho số mũ mới:  ∫f(x)dx = x{to_superscript(str(new_power))} / {human_math(new_power)}",
        "",
        f"Bước 4 — Rút gọn:  ∫f(x)dx = {final_answer}",
        "",
        f"Kết quả cuối cùng:  ∫f(x)dx = {final_answer} + C",
    ]


# --- Tra bảng nguyên hàm cơ bản (sin, cos, exp, ...) -----------------------

_TABLE_LOOKUP: dict[str, str] = {
    "sin":  "∫sin(x)dx = −cos(x) + C",
    "cos":  "∫cos(x)dx = sin(x) + C",
    "tan":  "∫tan(x)dx = −ln|cos(x)| + C",
    "exp":  "∫eˣdx = eˣ + C",
    "sinh": "∫sinh(x)dx = cosh(x) + C",
    "cosh": "∫cosh(x)dx = sinh(x) + C",
    "sqrt": "∫√x dx = (2/3)x^(3/2) + C",
}


def _is_simple_table_lookup(expr: Expr, x: Symbol) -> bool:
    return (
        hasattr(expr, "func")
        and len(getattr(expr, "args", ())) == 1
        and expr.args[0] == x
        and expr.func.__name__ in _TABLE_LOOKUP
    )


def _table_lookup_working(expr: Expr, antiderivative: Expr, x: Symbol) -> list[str]:
    func_name = expr.func.__name__
    formula = _TABLE_LOOKUP[func_name]
    final_answer = human_math(_safe_expand(antiderivative))
    return [
        "Các bước tính nguyên hàm:",
        "Quy tắc sử dụng: TRA BẢNG NGUYÊN HÀM CƠ BẢN",
        "",
        f"f(x) = {human_math(expr)} là một hàm số cơ bản có nguyên hàm đã biết:",
        "",
        f"  {formula}",
        "",
        f"Kết quả cuối cùng:  ∫f(x)dx = {final_answer} + C",
    ]


# --- Đổi biến số (u-substitution / quy tắc chuỗi đảo ngược) ----------------

def _is_chain_like(expr: Expr, x: Symbol) -> bool:
    return (
        hasattr(expr, "func")
        and len(getattr(expr, "args", ())) == 1
        and expr.args[0] != x
        and expr.args[0].has(x)
    )


def _substitution_working(expr: Expr, antiderivative: Expr, x: Symbol) -> list[str] | None:
    """Detect f(g(x))·g'(x)-style integrands where u = g(x) simplifies things."""
    # Case A: expr itself is h(g(x)) with g(x) != x (pure chain-like, no
    # extra multiplicative factor — e.g. sin(x**2) alone is NOT solvable by
    # substitution without a matching g'(x) factor, but we still show the
    # substitution idea when g is linear (ax+b), since that's a very common
    # student pattern with an implicit constant factor.
    candidates: list[Expr] = []
    if expr.is_Mul:
        factors = [f for f in expr.args if not f.is_number]
        for i, factor in enumerate(factors):
            if _is_chain_like(factor, x):
                inside = factor.args[0]
                rest = Mul(*[f for j, f in enumerate(factors) if j != i]) if len(factors) > 1 else Integer(1)
                candidates.append((factor, inside, rest))
    elif _is_chain_like(expr, x):
        inside = expr.args[0]
        if inside.is_Add or (inside.is_Mul and inside.has(x)):
            # e.g. sin(3*x + 1) — inside is linear in x
            candidates.append((expr, inside, Integer(1)))

    for factor, inside, rest in candidates:
        inside_derivative = simplify(diff(inside, x))
        if inside_derivative == 0:
            continue
        # Check whether `rest` is a constant multiple of inside_derivative
        ratio = simplify(rest / inside_derivative) if inside_derivative != 0 else None
        if ratio is not None and ratio.is_number:
            return _build_substitution_lines(expr, antiderivative, x, factor, inside, inside_derivative, rest, ratio)
    return None


def _build_substitution_lines(expr, antiderivative, x, factor, inside, inside_derivative, rest, ratio) -> list[str]:
    u_str = human_math(inside)
    du_str = human_math(inside_derivative)
    final_answer = human_math(_safe_expand(antiderivative))
    outer_func = factor.func.__name__
    lines = [
        "Các bước tính nguyên hàm:",
        "Quy tắc sử dụng: PHƯƠNG PHÁP ĐỔI BIẾN SỐ (u-substitution)",
        "",
        "Dùng phương pháp này khi biểu thức có dạng f(g(x))·g′(x).",
        "",
        f"f(x) = {human_math(expr)}",
        "",
        "Bước 1 — Đặt biến số mới:",
        f"  u = {u_str}",
        "",
        "Bước 2 — Lấy đạo hàm của u theo x:",
        f"  du/dx = {du_str}   ⟹   du = ({du_str}) dx",
        "",
    ]
    if ratio == 1:
        lines.extend([
            "Bước 3 — Nhận thấy phần còn lại của biểu thức chính là du:",
            f"  f(x)dx  =  {outer_func}(u) du",
        ])
    else:
        lines.extend([
            f"Bước 3 — Phần còn lại của biểu thức bằng {human_math(ratio)}·du, nên:",
            f"  f(x)dx  =  {human_math(ratio)}·{outer_func}(u) du",
        ])
    lines.extend([
        "",
        "Bước 4 — Tích phân theo biến u (dùng bảng nguyên hàm cơ bản):",
        f"  ∫ ... du  →  thay ngược u = {u_str}",
        "",
        f"Kết quả cuối cùng:  ∫f(x)dx = {final_answer} + C",
    ])
    return lines


# --- Tích phân từng phần (integration by parts) ----------------------------

def _by_parts_working(expr: Expr, antiderivative: Expr, x: Symbol) -> list[str] | None:
    factors = [f for f in expr.args if not f.is_number]
    if len(factors) < 2:
        return None
    # Prefer picking a polynomial factor as "u" (LIATE-ish heuristic)
    poly_factors = [f for f in factors if f.is_polynomial(x) and not f.is_number]
    if not poly_factors:
        return None
    u = poly_factors[0]
    dv_expr = simplify(expr / u)
    try:
        v = simplify(integrate(dv_expr, x))
        if v.has(Integral):
            return None
    except Exception:
        return None
    du = simplify(diff(u, x))
    remaining = simplify(v * du)
    try:
        remaining_integral = simplify(integrate(remaining, x))
        if remaining_integral.has(Integral):
            return None
    except Exception:
        return None
    final_answer = human_math(_safe_expand(antiderivative))
    return [
        "Các bước tính nguyên hàm:",
        "Quy tắc sử dụng: TÍCH PHÂN TỪNG PHẦN (Integration by Parts)",
        "",
        "Công thức:  ∫u dv = uv − ∫v du",
        "Dùng quy tắc này khi hai loại hàm khác nhau được nhân với nhau",
        "(ví dụ: đa thức × lượng giác, đa thức × mũ).",
        "",
        "Bước 1 — Chọn u (nên chọn phần đơn giản hoá khi lấy đạo hàm) và dv:",
        f"  u  = {human_math(u)}",
        f"  dv = {human_math(dv_expr)} dx",
        "",
        "Bước 2 — Tính du (đạo hàm u) và v (nguyên hàm dv):",
        f"  du = {human_math(du)} dx",
        f"  v  = {human_math(v)}",
        "",
        "Bước 3 — Thay vào công thức từng phần:",
        f"  ∫f(x)dx = ({human_math(u)})({human_math(v)}) − ∫ ({human_math(v)})({human_math(du)}) dx",
        "",
        "Bước 4 — Tính tích phân còn lại và rút gọn:",
        f"  ∫f(x)dx = {final_answer} + C",
        "",
        f"Kết quả cuối cùng:  ∫f(x)dx = {final_answer} + C",
    ]


# --- Trường hợp mặc định ----------------------------------------------------

def _generic_working(expr: Expr, antiderivative: Expr) -> list[str]:
    final_answer = human_math(_safe_expand(antiderivative))
    note = ""
    if antiderivative.has(Integral):
        note = (
            "\nGhi chú: SymPy không tìm được dạng sơ cấp cho tích phân này — "
            "kết quả được để dưới dạng tích phân chưa giải (Integral)."
        )
    return [
        "Các bước tính nguyên hàm:",
        "Quy tắc sử dụng: TÍNH TRỰC TIẾP (SymPy)",
        "",
        f"f(x)      = {human_math(expr)}",
        f"∫f(x)dx   = {final_answer} + C{note}",
    ]
