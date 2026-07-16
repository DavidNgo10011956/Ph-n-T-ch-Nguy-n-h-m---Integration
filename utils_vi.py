"""General utilities: colour codes, safe numeric evaluation, and input helpers."""

from __future__ import annotations

import re

import numpy as np
import sympy as sp

try:
    import scipy.special as _sc
except ImportError:
    _sc = None

# ---------------------------------------------------------------------------
# Terminal colour codes (graceful fallback if colorama is not installed)
# ---------------------------------------------------------------------------

try:
    import colorama
    colorama.init(autoreset=True)
    C_RESET   = colorama.Style.RESET_ALL
    C_BOLD    = colorama.Style.BRIGHT
    C_CYAN    = colorama.Fore.CYAN
    C_GREEN   = colorama.Fore.GREEN
    C_YELLOW  = colorama.Fore.YELLOW
    C_RED     = colorama.Fore.RED
    C_BLUE    = colorama.Fore.BLUE
    C_MAGENTA = colorama.Fore.MAGENTA
except ImportError:
    C_RESET = C_BOLD = C_CYAN = C_GREEN = ""
    C_YELLOW = C_RED = C_BLUE = C_MAGENTA = ""


# ---------------------------------------------------------------------------
# Superscript / subscript helpers
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = str.maketrans(
    "0123456789-+().abcdefghijklmnopqrstuvwxyz",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾·ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖ𐞥ʳˢᵗᵘᵛʷˣʸᶻ",
)

_SUBSCRIPT_MAP = str.maketrans(
    "0123456789-+().",
    "₀₁₂₃₄₅₆₇₈₉₋₊₍₎.",
)


def to_superscript(value: str | int) -> str:
    """Translate digit / sign characters to Unicode superscripts."""
    return str(value).translate(_SUPERSCRIPT_MAP)


def to_subscript(value: str | int) -> str:
    """Translate digit / sign characters to Unicode subscripts."""
    return str(value).translate(_SUBSCRIPT_MAP)


# ---------------------------------------------------------------------------
# Visual distinction for the variable x (matplotlib only)
# ---------------------------------------------------------------------------
#
# human_math() renders multiplication as the "×" sign, but a plain ASCII
# "x" used for the variable can still look confusingly similar to it at a
# glance (same width/shape in many sans-serif fonts). A true italic Unicode
# letter ("𝑥", U+1D465) would fix that, but that glyph isn't included in
# Matplotlib's default font (DejaVu Sans) and renders as a missing-glyph
# box. Instead we lean on Matplotlib's own mathtext engine: wrapping just
# the variable in $...$ makes it render in italics (the standard
# typographic convention for a variable) using a font Matplotlib already
# ships with, while leaving every other character — including the ×
# multiplication sign — untouched.
_STANDALONE_X_RE = re.compile(r"(?<![A-Za-z_])x(?![A-Za-z0-9_])")


def italicize_x(text: str) -> str:
    """Wrap every standalone variable 'x' in $x$ for italic mathtext rendering.

    Only for use in Matplotlib text (titles, labels, legends, annotations).
    Skips any 'x' that's part of a longer word (e.g. "max", "next") so
    those are left completely untouched.
    """
    return _STANDALONE_X_RE.sub(r"$x$", text)


# ---------------------------------------------------------------------------
# Internal helpers for human_math
# ---------------------------------------------------------------------------

def _to_full_superscript(s: str) -> str:
    """Superscript every character of an already-simplified exponent body
    (used inside a parenthesised compound exponent, where the whole thing
    is already 'raised', so letters/digits/operators all go to superscript)."""
    s = s.replace("^", "").replace("*", "")
    return "".join(to_superscript(ch) if (ch.isalnum() or ch in "-+().") else ch for ch in s)


def _exp_superscript(exponent: str) -> str:
    """Convert an exp(…) argument into a compact e^… Unicode form."""
    e = exponent.strip().replace(" ", "").replace("**", "^")

    # Bare single letter: exp(x) → eˣ
    if re.fullmatch(r"[a-z]", e):
        return "e" + to_superscript(e)

    # Whole exponent is a parenthesised group raised to a power,
    # e.g. (x^2+1)^2  →  e⁽ˣ²⁺¹⁾²   (common in erf/erfc/erfi derivatives)
    whole_group = re.fullmatch(r"\((.+)\)\^(-?\d+)", e)
    if whole_group:
        inner_sup = _to_full_superscript(whole_group.group(1))
        power_sup = to_superscript(whole_group.group(2))
        return "e" + "⁽" + inner_sup + "⁾" + power_sup

    # [sign][coeff]*letter^power  e.g. -2*x^2 → ⁻²x²
    def _coeff_var_pow(m):
        sign  = "⁻" if m.group(1) == "-" else (to_superscript(m.group(1)) if m.group(1) else "")
        coeff = to_superscript(m.group(2)) if m.group(2) else ""
        var   = to_superscript(m.group(3))
        power = to_superscript(m.group(4))
        return sign + coeff + var + power

    e = re.sub(r"(-?)(\d*)\*?([a-z])\^-?(\d+)", _coeff_var_pow, e)

    # [sign][coeff]*letter (bare)
    def _coeff_var(m):
        sign  = "⁻" if m.group(1) == "-" else (to_superscript(m.group(1)) if m.group(1) else "")
        coeff = to_superscript(m.group(2)) if m.group(2) else ""
        var   = to_superscript(m.group(3))
        return sign + coeff + var

    e = re.sub(r"(?<![⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺])(-?)(\d*)\*?([a-z])(?!\^)", _coeff_var, e)

    # If the exponent is too complex to cleanly superscript (e.g. a product
    # of multiple grouped terms like x^2*(x-3)^2), it still contains a raw
    # '*' or '^' here. Fall back to plain e^(...) notation rather than a
    # partially-converted, hard-to-read mix of Unicode and literal symbols.
    if "*" in e or "^" in e:
        original = exponent.strip().replace(" ", "").replace("**", "^")
        return f"e^({original})"

    # Remaining digits / signs / punctuation
    def _sup_non_letter(s: str) -> str:
        return "".join(ch if ch.isalpha() else to_superscript(ch) for ch in s)

    parts = e.split("/")
    return "e" + "/".join(_sup_non_letter(p) for p in parts)


def _sympy_log_base(expr) -> str:
    """
    Walk a SymPy expression tree and convert log(x)/log(b) ratios
    (the way SymPy internally stores log(x, b)) into "log(x, b)" strings
    so the regex stage can later format them as log_b(x).
    """
    def _walk(e):
        if e.is_Mul:
            num_logs, den_logs, others = [], [], []
            for arg in e.args:
                if isinstance(arg, sp.log):
                    num_logs.append(arg.args[0])
                elif (arg.is_Pow
                      and isinstance(arg.base, sp.log)
                      and arg.exp == sp.Integer(-1)):
                    den_logs.append(arg.base.args[0])
                else:
                    others.append(arg)
            if len(num_logs) == 1 and len(den_logs) == 1:
                base_str = str(den_logs[0])
                arg_str  = _walk(num_logs[0])
                token = f"log({arg_str}, {base_str})"
                if others:
                    return f"{_walk(sp.Mul(*others))}*{token}"
                return token
        return str(e)

    return _walk(expr)


# ---------------------------------------------------------------------------
# Function name lists (longer names must precede shorter prefixes)
# ---------------------------------------------------------------------------

_TRIG = [
    "sinh", "cosh", "tanh", "sech", "csch", "coth",
    "asinh", "acosh", "atanh",
    "asin", "acos", "atan",
    "sin", "cos", "tan", "sec", "csc", "cot",
]
_LOG_FUNCS = ["log10", "log2", "log", "ln"]
_SPECIAL = ["erfc", "erfi", "erf", "sinc", "gamma", "zeta", "Heaviside", "DiracDelta"]
_ALL_FUNCS = _TRIG + _LOG_FUNCS + _SPECIAL

# Display aliases for inverse-trig and special functions
_FUNC_DISPLAY: dict[str, str] = {
    "asin":  "arcsin",  "acos":  "arccos",  "atan":  "arctan",
    "asinh": "arcsinh", "acosh": "arccosh", "atanh": "arctanh",
    "DiracDelta": "δ",
    "Heaviside":  "H",
    "gamma":      "Γ",
    "zeta":       "ζ",
}


# ---------------------------------------------------------------------------
# Helpers for unevaluated Derivative(...) / Subs(Derivative(...), ...) nodes
# ---------------------------------------------------------------------------
#
# SymPy leaves diff() results unevaluated for functions with no symbolic
# derivative rule (currently just zeta in this app). Simple cases look like
# Derivative(zeta(x), x); chain-rule cases (e.g. d/dx zeta(x**2)) look like
# Subs(Derivative(zeta(_xi_1), _xi_1), _xi_1, x**2) — "the derivative of
# zeta, evaluated at x**2". A plain regex can't safely extract the trailing
# substituted value when it may itself contain parentheses or when several
# such terms appear side-by-side with no comma between them (e.g. combined
# via "+"), so we parse parenthesis-balance explicitly instead.

_PRIME_MARKS: dict[int, str] = {1: "′", 2: "″", 3: "‴"}


def _match_parens(s: str, open_index: int) -> int:
    """Return the index of the ')' matching the '(' at s[open_index], or -1."""
    depth = 0
    for i in range(open_index, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are not nested inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _render_named_derivative(func_name: str, order: int, arg: str) -> str:
    display_name = _FUNC_DISPLAY.get(func_name, func_name)
    mark = _PRIME_MARKS.get(order, to_superscript(f"({order})"))
    return f"{display_name}{mark}({arg.strip()})"


_DERIVATIVE_INNER_RE = re.compile(
    r"^Derivative\((\w+)\(_xi_\d+\),\s*(?:\(_xi_\d+,\s*(\d+)\)|_xi_\d+)\)$"
)


def _parse_subs_of_derivative(content: str) -> str | None:
    """Parse the inside of Subs(Derivative(f(_xi_N), ...), _xi_N, VALUE).

    ``content`` is everything between the outer 'Subs(' and its matching
    ')'. Returns the f′(VALUE)-style rendering, or None if the content
    doesn't match this shape (left untouched by the caller in that case).
    """
    parts = _split_top_level(content)
    if len(parts) != 3:
        return None
    deriv_part, _var_part, value_part = (p.strip() for p in parts)
    m = _DERIVATIVE_INNER_RE.match(deriv_part)
    if not m:
        return None
    func_name = m.group(1)
    order = int(m.group(2)) if m.group(2) else 1
    return _render_named_derivative(func_name, order, value_part)


def _replace_subs_derivatives(text: str) -> str:
    """Find every top-level Subs(...) call and replace Subs(Derivative(...))
    patterns with clean f′(x)-style notation, leaving anything else as-is."""
    out: list[str] = []
    idx = 0
    while True:
        m = re.search(r"\bSubs\(", text[idx:])
        if not m:
            out.append(text[idx:])
            break
        start = idx + m.start()
        open_paren = idx + m.end() - 1
        close_paren = _match_parens(text, open_paren)
        if close_paren == -1:
            out.append(text[idx:])
            break
        content = text[open_paren + 1:close_paren]
        replacement = _parse_subs_of_derivative(content)
        out.append(text[idx:start])
        out.append(replacement if replacement is not None else text[start:close_paren + 1])
        idx = close_paren + 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Main human_math function
# ---------------------------------------------------------------------------

def human_math(expr) -> str:
    """
    Convert a SymPy expression (or plain string) into student-friendly Unicode text.

    Supported transformations
    -------------------------
    Powers          x**4       →  x⁴,   x**(-2)   →  x⁻²
    Exp             exp(x)     →  eˣ,   exp(2*x)  →  e²ˣ
    Trig powers     sin(x)**2  →  sin²(x)
    Roots           sqrt(x)    →  √x,   cbrt(x)   →  ∛x
    Log bases       log(x, 2)  →  log₂(x)
    Absolute value  Abs(x)     →  |x|
    Inverse trig    asin(x)    →  arcsin(x)
    Special funcs   gamma(x)   →  Γ(x),  DiracDelta(x) → δ(x)
    """
    # 1. Canonicalise via SymPy; intercept log-base before str()
    try:
        # sp.cancel()'s internal đa thức conversion auto-distributes
        # exp(a+b) -> e^a * e^b whenever the exponent is a compound
        # expression (e.g. exp((x**2+1)**2) -> E*exp(2*x**2)*exp(x**4)),
        # which is much harder to read than the original. sp.simplify()
        # still cancels common rational factors but leaves exp() alone.
        from sympy import exp as _exp_cls
        has_exp = bool(expr.atoms(_exp_cls)) if hasattr(expr, "atoms") else False
        expr = sp.simplify(expr) if has_exp else sp.cancel(expr)
        text = _sympy_log_base(expr)
    except Exception:
        text = str(expr)

    # 1b. Unevaluated Derivative(...) / Subs(Derivative(...)) nodes.
    # SymPy leaves diff() results unevaluated for functions with no
    # symbolic derivative rule (e.g. zeta), so diff(zeta(x), x) stays as
    # Derivative(zeta(x), x), and chain-rule cases like d/dx zeta(x**2)
    # become Subs(Derivative(zeta(_xi_1), _xi_1), _xi_1, x**2). Render
    # both as clean ζ′(x) / ζ′(x²) style notation instead of leaking
    # SymPy internals to the user.
    text = _replace_subs_derivatives(text)

    def _format_derivative(match: re.Match) -> str:
        func_name = match.group(1)
        arg = match.group(2).strip()
        order_str = match.group(3) if match.lastindex and match.lastindex >= 3 else None
        order = int(order_str) if order_str else 1
        return _render_named_derivative(func_name, order, arg)

    text = re.sub(
        r"\bDerivative\(([A-Za-z_]+)\(([^()]+)\),\s*\(x,\s*(\d+)\)\)",
        _format_derivative, text,
    )
    text = re.sub(
        r"\bDerivative\(([A-Za-z_]+)\(([^()]+)\),\s*x\)",
        _format_derivative, text,
    )
    # DiracDelta(x, n) is SymPy's own notation for the n-th derivative of
    # the delta function (it doesn't go through the Derivative(...) wrapper).
    text = re.sub(
        r"\bDiracDelta\(([^,()]+),\s*(\d+)\)",
        lambda m: f"δ{_PRIME_MARKS.get(int(m.group(2)), to_superscript(f'({m.group(2)})'))}({m.group(1).strip()})",
        text,
    )

    text = text.replace("**", "^")

    # 2. exp(…) → eˢᵘᵖ
    text = re.sub(r"\bexp\(((?:[^()]|\([^()]*\))+)\)", lambda m: _exp_superscript(m.group(1)), text)
    text = re.sub(r"\bE\^\(?([^()^]+)\)?", lambda m: _exp_superscript(m.group(1)), text)

    # 3. Roots
    text = re.sub(
        r"\broot\(([^,()]+),\s*(\d+)\)",
        lambda m: to_superscript(m.group(2)) + "√" + m.group(1).strip(), text,
    )
    text = re.sub(r"\bcbrt\(([^()]+)\)", lambda m: "∛" + m.group(1).strip(), text)
    text = re.sub(
        r"\bsqrt\(([^()]+)\)\^\(?(-?\d+)\)?",
        lambda m: "√" + m.group(1).strip() + to_superscript(m.group(2)), text,
    )
    text = re.sub(r"\bsqrt\(([^()]+)\)", lambda m: "√" + m.group(1).strip(), text)

    # 4. Log with explicit base: log(x, b) → logᵦ(x)
    text = re.sub(
        r"\blog\(([^,()]+),\s*(\d+)\)",
        lambda m: "log" + to_subscript(m.group(2)) + "(" + m.group(1).strip() + ")", text,
    )
    text = re.sub(r"\blog2\(([^()]+)\)",  lambda m: "log₂(" + m.group(1).strip() + ")", text)
    text = re.sub(r"\blog10\(([^()]+)\)", lambda m: "log₁₀(" + m.group(1).strip() + ")", text)

    # 4b. polygamma(n, x) → ψ(x)  [n=0, digamma]  or  ψ⁽ⁿ⁾(x)  [n>0]
    # This appears automatically when SymPy differentiates gamma(x).
    text = re.sub(
        r"\bpolygamma\(0,\s*([^,()]+)\)",
        lambda m: f"ψ({m.group(1).strip()})", text,
    )
    text = re.sub(
        r"\bpolygamma\((\d+),\s*([^,()]+)\)",
        lambda m: f"ψ{to_superscript('(' + m.group(1) + ')')}({m.group(2).strip()})", text,
    )
    text = re.sub(r"\bloggamma\(([^()]+)\)", lambda m: f"lnΓ({m.group(1).strip()})", text)

    # 5. f(x)^n → fⁿ(x)  or  1/f|n|(x) for negative powers
    for func in _ALL_FUNCS:
        display = _FUNC_DISPLAY.get(func, func)
        text = re.sub(
            rf"\b{func}\(([^()]+)\)\^\(?(-?\d+)\)?",
            lambda m, d=display: (
                f"1/{d}{to_superscript(abs(int(m.group(2))))}"
                f"({m.group(1).strip()})"
                if int(m.group(2)) < 0
                else f"{d}{to_superscript(m.group(2))}({m.group(1).strip()})"
            ),
            text,
        )

    # 6. Rename functions to display forms
    for orig, pretty in _FUNC_DISPLAY.items():
        text = re.sub(rf"\b{re.escape(orig)}\b", pretty, text)

    # 7. Variable powers: x^n, y^n, …
    # NOTE: do NOT use \(? … \)? here — it would eat closing brackets belonging
    # to enclosing functions like cos(x^2).
    text = re.sub(r"\bx\^(-?\d+)", lambda m: "x" + to_superscript(m.group(1)), text)
    text = re.sub(r"\b([a-wyzA-Z])\^(-?\d+)",
                  lambda m: m.group(1) + to_superscript(m.group(2)), text)

    # 8. Absolute value, ceiling, floor
    text = re.sub(r"\bAbs\(([^()]+)\)",     lambda m: f"|{m.group(1).strip()}|", text)
    text = re.sub(r"\bceiling\(([^()]+)\)", lambda m: f"⌈{m.group(1).strip()}⌉", text)
    text = re.sub(r"\bfloor\(([^()]+)\)",   lambda m: f"⌊{m.group(1).strip()}⌋", text)

    # 9. Factorial, complex helpers, Max/Min
    text = re.sub(r"\bfactorial\(([^()]+)\)", lambda m: f"{m.group(1).strip()}!", text)
    text = re.sub(r"\bre\(([^()]+)\)",   lambda m: f"Re({m.group(1).strip()})", text)
    text = re.sub(r"\bim\(([^()]+)\)",   lambda m: f"Im({m.group(1).strip()})", text)
    text = re.sub(r"\bconjugate\(([^()]+)\)",
                  lambda m: f"{m.group(1).strip()}\u0305", text)
    text = re.sub(r"\bsign\(([^()]+)\)", lambda m: f"sgn({m.group(1).strip()})", text)
    text = re.sub(r"\bMax\(", "max(", text)
    text = re.sub(r"\bMin\(", "min(", text)

    # 10. Multiplication cleanup
    text = text.replace("*", " × ")

    # Implicit multiplication: digit × letter / function / e / √
    text = re.sub(r"(\d) × ([a-zA-Z])", r"\1\2", text)
    func_pattern = "|".join(re.escape(_FUNC_DISPLAY.get(f, f)) for f in _ALL_FUNCS)
    text = re.sub(rf"(\d) × ({func_pattern})\b", r"\1\2", text)
    text = re.sub(r"\) × (e[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ˣ])", r")\1", text)
    text = re.sub(r"([\d⁰¹²³⁴⁵⁶⁷⁸⁹]) × e([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ˣ])", r"\1e\2", text)
    text = re.sub(r"(\d) × e\b", r"\1e", text)
    text = re.sub(r"(\d) × √", r"\1√", text)
    text = re.sub(r"([a-zA-Z⁰¹²³⁴⁵⁶⁷⁸⁹]) × (e[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ˣ])", r"\1\2", text)

    # 10b. Greek / math constants
    text = re.sub(r"\bpi\b", "π", text)
    # Standalone Euler's e (not inside eˣ notation)
    text = re.sub(r"\bE\b(?![⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ˣ])", "e", text)

    # 11. Sign cleanup
    text = text.replace("+ -", "- ")
    text = text.replace("- -", "+ ")
    text = text.replace(" × -", " - ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Numeric backend for special functions (NumPy alone doesn't know these)
# ---------------------------------------------------------------------------
#
# lambdify(..., modules=["numpy"]) fails at *call time* (not at lambdify time)
# for gamma, zeta, erf, erfi, sinc, Heaviside, DiracDelta and polygamma
# (which SymPy introduces automatically when differentiating gamma(x)),
# because none of these exist in the numpy namespace. That produced the
# "error" the user saw whenever plotting / root-finding touched a special
# function. We provide explicit, vectorised implementations instead.

def _heaviside_np(value, at_zero=0.5):
    # SymPy's Heaviside(x, h0) takes an optional second argument specifying
    # its value exactly at x = 0 (defaults to 1/2).
    return np.heaviside(value, at_zero)


def _diracdelta_np(value, order=0):
    # DiracDelta(x, n) is the n-th distributional derivative of the delta
    # function. As a distribution it is zero everywhere except at x = 0
    # (where it is not an ordinary finite value), so for plotting /
    # numerical purposes every order evaluates as zero almost everywhere.
    arr = np.asarray(value, dtype=float)
    return np.zeros_like(arr)


def _sinc_np(value):
    # SymPy's sinc(x) = sin(x)/x (unnormalised). NumPy's np.sinc is the
    # *normalised* sin(pi x)/(pi x), so it cannot be reused directly.
    arr = np.asarray(value, dtype=float)
    with np.errstate(all="ignore"):
        result = np.sin(arr) / np.where(arr == 0, 1.0, arr)
    return np.where(np.abs(arr) < 1e-12, 1.0, result)


def _zeta_np(value):
    if _sc is None:
        return np.full_like(np.asarray(value, dtype=float), np.nan)
    arr = np.asarray(value, dtype=float)
    with np.errstate(all="ignore"):
        # SymPy's single-argument zeta(x) is the Riemann zeta function,
        # equivalent to the Hurwitz zeta function with q = 1.
        return _sc.zeta(arr, 1)


def _polygamma_np(order, value):
    if _sc is None:
        return np.full_like(np.asarray(value, dtype=float), np.nan)
    return _sc.polygamma(int(order), value)


_SPECIAL_NUMERIC_MAP: dict[str, object] = {
    "sinc":       _sinc_np,
    "Heaviside":  _heaviside_np,
    "DiracDelta": _diracdelta_np,
    "zeta":       _zeta_np,
    "polygamma":  _polygamma_np,
}
if _sc is not None:
    _SPECIAL_NUMERIC_MAP.update({
        "erf":      _sc.erf,
        "erfc":     _sc.erfc,
        "erfi":     _sc.erfi,
        "gamma":    _sc.gamma,
        "loggamma": _sc.gammaln,
    })

# Passed straight to sympy.lambdify's `modules=` argument. Custom special
# functions are tried first, then plain NumPy for everything else.
LAMBDIFY_MODULES = [_SPECIAL_NUMERIC_MAP, "numpy"]


def _fd_numeric(base_numeric_func, order: int):
    """Build a numeric finite-difference derivative of a base numeric function.

    Used as a fallback when SymPy has no symbolic derivative rule for a
    special function (e.g. zeta), so the result stays an unevaluated
    Derivative(...) node that ordinary lambdify cannot turn into code.
    """
    def _fd(value):
        arr = np.asarray(value, dtype=float)
        h = 1e-4
        with np.errstate(all="ignore"):
            if order <= 1:
                return (base_numeric_func(arr + h) - base_numeric_func(arr - h)) / (2 * h)
            if order == 2:
                return (
                    base_numeric_func(arr + h) - 2 * base_numeric_func(arr) + base_numeric_func(arr - h)
                ) / (h ** 2)
            # Higher orders: repeated central differencing (numerically rough
            # but keeps the app from crashing; rarely requested in practice).
            def _nth(v, n):
                if n == 0:
                    return base_numeric_func(v)
                return (_nth(v + h, n - 1) - _nth(v - h, n - 1)) / (2 * h)
            return _nth(arr, order)
    return _fd


def safe_lambdify(x, expr):
    """lambdify() an expression with numeric support for special functions.

    Use this everywhere instead of calling sympy.lambdify directly with
    modules=["numpy"], so gamma(x), zeta(x), erf(x), erfi(x), sinc(x),
    Heaviside(x), DiracDelta(x) and polygamma(...) (which appears when
    differentiating gamma) all evaluate correctly.

    Some special functions (notably zeta) have no symbolic derivative rule
    in SymPy, so diff() leaves an unevaluated Derivative(...) node behind.
    Those are replaced here with a numeric finite-difference fallback so
    evaluation still succeeds instead of raising at call time.
    """
    from sympy import Derivative, Function, lambdify

    working_expr = expr
    extra_map: dict[str, object] = {}
    for i, deriv in enumerate(expr.atoms(Derivative)):
        base = deriv.expr
        func_name = getattr(getattr(base, "func", None), "__name__", None)
        base_numeric = _SPECIAL_NUMERIC_MAP.get(func_name)
        if base_numeric is None:
            continue
        order = int(getattr(deriv, "derivative_count", len(deriv.variables)))
        placeholder_name = f"_fd_{func_name}_{order}_{i}"
        placeholder = Function(placeholder_name)(x)
        extra_map[placeholder_name] = _fd_numeric(base_numeric, order)
        working_expr = working_expr.xreplace({deriv: placeholder})

    modules = [extra_map, *LAMBDIFY_MODULES] if extra_map else LAMBDIFY_MODULES
    return lambdify(x, working_expr, modules=modules)


# ---------------------------------------------------------------------------
# Safe numeric evaluation
# ---------------------------------------------------------------------------

def safe_eval(func, x_values: np.ndarray) -> np.ndarray:
    """Evaluate a lambdified function safely, replacing non-finite values with NaN.

    Also clips extreme spikes (e.g. near vertical asymptotes) so the plot
    y-axis stays sensible.
    """
    with np.errstate(all="ignore"):
        y_values = func(x_values)
    if np.isscalar(y_values):
        y_values = np.full_like(x_values, float(y_values), dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    y_values = np.where(np.isfinite(y_values), y_values, np.nan)
    finite = np.isfinite(y_values)
    if finite.sum() > 2:
        median_abs = np.nanmedian(np.abs(y_values))
        if median_abs > 0:
            spike_threshold = max(median_abs * 100, 1e4)
            jumps = np.abs(np.diff(y_values, prepend=y_values[0]))
            y_values[jumps > spike_threshold] = np.nan
    return y_values


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def ask_int(prompt: str, default: int, minimum: int, maximum: int) -> int:
    """Ask the user for an integer with a default and range validation."""
    while True:
        raw = input(f"{C_GREEN}{prompt} [{default}]: {C_RESET}").strip() or str(default)
        try:
            value = int(raw)
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"{C_YELLOW}Please enter a whole number between {minimum} and {maximum}.{C_RESET}")


def ask_float(prompt: str, default: float) -> float:
    """Ask the user for a floating-point number with a default."""
    while True:
        raw = input(f"{C_GREEN}{prompt} [{default}]: {C_RESET}").strip() or str(default)
        try:
            return float(raw)
        except ValueError:
            print(f"{C_YELLOW}Please enter a valid number (e.g. -6, 0, 3.5).{C_RESET}")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question and return a boolean."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{C_GREEN}{prompt} {hint}: {C_RESET}").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print(f"{C_YELLOW}Please type y (yes) or n (no).{C_RESET}")