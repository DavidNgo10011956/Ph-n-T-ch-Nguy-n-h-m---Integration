"""Numerical root finding and special point classification — integral edition.

Key insight this module leans on: if F(x) is an antiderivative of f(x)
(F' = f), then:
  - F(x) has a critical point (local max/min) exactly where f(x) = 0
    and f changes sign there — i.e. the ROOTS of f(x) itself.
  - F(x) has an inflection point exactly where f'(x) = 0 and f' changes
    sign there — i.e. the roots of the derivative of f.

So the exact same sign-change root-finding machinery used for derivative
analysis works here unchanged; we simply feed it f(x) in the role
previously played by "the first derivative", and f'(x) in the role
previously played by "the second derivative". `find_special_points_of_antiderivative`
below is a thin, clearly-named wrapper that does this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sympy import Expr, Symbol, solve

from utils_vi import safe_eval, safe_lambdify

try:
    from scipy import optimize as scipy_opt
except ImportError:
    scipy_opt = None

SpecialPoint   = tuple[float, float, str]
InflectionPoint = tuple[float, float]


@dataclass(frozen=True)
class AnalysisConfig:
    n_root_points:    int                = 9000
    tangent_threshold: float             = 1e-3
    merge_tol_factor: float              = 1e-3
    zero_snap_factor: float              = 1e-3
    sign_steps:       tuple[float, ...]  = (2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2)


CFG = AnalysisConfig()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_special_points(
    expr: Expr,
    derivatives: dict[int, Expr],
    x: Symbol,
    x_min: float,
    x_max: float,
) -> tuple[list[SpecialPoint], list[InflectionPoint]]:
    """Find critical points and inflection points within the given x-range."""
    critical   = _find_critical_points(expr, derivatives.get(1), x, x_min, x_max)
    inflection = _find_inflection_points(expr, derivatives.get(2), x, x_min, x_max)
    return critical, inflection


def find_special_points_of_antiderivative(
    f_expr: Expr,
    antiderivative: Expr,
    x: Symbol,
    x_min: float,
    x_max: float,
) -> tuple[list[SpecialPoint], list[InflectionPoint]]:
    """Find critical / inflection points of F(x), the antiderivative of f(x).

    Since F' = f, F's critical points are f's roots, and F's inflection
    points are f''s (f-prime's) roots. This re-uses find_special_points by
    handing it F as the "function" (for y-values) and {1: f, 2: f'} as the
    "derivatives" dict (for locating the roots).
    """
    from sympy import diff, simplify
    f_prime = simplify(diff(f_expr, x))
    pseudo_derivatives = {1: f_expr, 2: f_prime}
    return find_special_points(antiderivative, pseudo_derivatives, x, x_min, x_max)


def find_x_intercepts(f_expr: Expr, x: Symbol, x_min: float, x_max: float) -> list[tuple[float, float]]:
    """Find where f(x) crosses zero — these bound regions of signed area
    and are exactly the critical-point x-values of F(x)."""
    roots = numerical_roots(f_expr, x, x_min, x_max)
    return [(r, 0.0) for r in roots]


# ---------------------------------------------------------------------------
# Critical points (where f′ = 0)
# ---------------------------------------------------------------------------

def _find_critical_points(
    expr: Expr,
    first_derivative: Optional[Expr],
    x: Symbol,
    x_min: float,
    x_max: float,
) -> list[SpecialPoint]:
    if first_derivative is None:
        return []
    points: list[SpecialPoint] = []
    for x_value in numerical_roots(first_derivative, x, x_min, x_max):
        try:
            y_value = float(expr.subs(x, x_value).evalf())
            if not np.isfinite(y_value):
                continue
            left_sign, right_sign = sign_on_sides(first_derivative, x, x_value, x_min, x_max)
            points.append((x_value, y_value, _classify_critical_point(left_sign, right_sign)))
        except Exception:
            continue
    return _dedupe_special_points(points, x_min, x_max)


# ---------------------------------------------------------------------------
# Inflection points (where f″ = 0 AND f″ changes sign)
# ---------------------------------------------------------------------------

def _find_inflection_points(
    expr: Expr,
    second_derivative: Optional[Expr],
    x: Symbol,
    x_min: float,
    x_max: float,
) -> list[InflectionPoint]:
    if second_derivative is None:
        return []
    points: list[SpecialPoint] = []
    for x_value in numerical_roots(second_derivative, x, x_min, x_max):
        left_sign, right_sign = sign_on_sides(second_derivative, x, x_value, x_min, x_max)
        # A true inflection requires a sign change in f″
        if left_sign is None or right_sign is None or left_sign == right_sign:
            continue
        try:
            y_value = float(expr.subs(x, x_value).evalf())
            if np.isfinite(y_value):
                points.append((x_value, y_value, "Điểm uốn"))
        except Exception:
            continue
    return [(xv, yv) for xv, yv, _ in _dedupe_special_points(points, x_min, x_max)]


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------

def numerical_roots(
    sym_expr: Expr,
    x: Symbol,
    x_min: float,
    x_max: float,
) -> list[float]:
    """Find real roots of a symbolic expression in [x_min, x_max].

    Uses three complementary strategies:
    1. Sign-change bisection (catches most roots)
    2. Tangent / near-zero search (catches tangent roots like x² = 0)
    3. Symbolic solve (catches exact roots)
    """
    try:
        numeric_func = safe_lambdify(x, sym_expr)
    except Exception:
        return []

    x_values = np.linspace(x_min, x_max, CFG.n_root_points)
    y_values = safe_eval(numeric_func, x_values)

    roots: list[float] = []
    roots.extend(_roots_by_sign_change(numeric_func, x_values, y_values))
    roots.extend(_roots_by_tangent_search(numeric_func, x_values, y_values, x_min, x_max))
    roots.extend(_roots_by_symbolic_solve(sym_expr, x, x_min, x_max))

    zero_tol = _zero_tol(x_min, x_max)
    roots = [
        _snap_zero(r, zero_tol)
        for r in roots
        if x_min <= r <= x_max and _validate_root(numeric_func, r, tol=1e-4)
    ]
    return _dedupe_values(roots, x_min, x_max)


def _roots_by_sign_change(numeric_func, x_values: np.ndarray, y_values: np.ndarray) -> list[float]:
    roots: list[float] = []
    for i in range(len(x_values) - 1):
        y_l, y_r = y_values[i], y_values[i + 1]
        if not (np.isfinite(y_l) and np.isfinite(y_r)):
            continue
        if y_l == 0.0:
            roots.append(float(x_values[i]))
            continue
        if y_l * y_r >= 0:
            continue
        if scipy_opt is None:
            roots.append(float((x_values[i] + x_values[i + 1]) / 2))
            continue
        try:
            root = scipy_opt.brentq(
                lambda v: float(numeric_func(v)),
                x_values[i], x_values[i + 1],
                xtol=1e-12, maxiter=200,
            )
            roots.append(float(root))
        except Exception:
            continue
    return roots


def _roots_by_tangent_search(numeric_func, x_values: np.ndarray, y_values: np.ndarray,
                              x_min: float, x_max: float) -> list[float]:
    """Find roots where the function barely touches zero without crossing (tangent roots)."""
    if scipy_opt is None:
        return []
    abs_y = np.abs(y_values)
    candidate_idxs = [
        i for i in range(1, len(x_values) - 1)
        if _is_local_abs_minimum(abs_y, i)
    ]
    roots: list[float] = []
    for cluster in _cluster_indexes(candidate_idxs, x_values, _merge_tol(x_min, x_max)):
        left  = x_values[max(0, cluster[0] - 2)]
        right = x_values[min(len(x_values) - 1, cluster[-1] + 2)]
        try:
            result = scipy_opt.minimize_scalar(
                lambda v: abs(float(numeric_func(v))),
                bounds=(left, right), method="bounded",
            )
            if result.success and _validate_root(numeric_func, float(result.x), tol=1e-5):
                roots.append(float(result.x))
        except Exception:
            continue
    return roots


def _roots_by_symbolic_solve(sym_expr: Expr, x: Symbol, x_min: float, x_max: float) -> list[float]:
    """Attempt an exact symbolic solve and filter to the current range."""
    roots: list[float] = []
    try:
        for root in solve(sym_expr, x):
            if root.is_real:
                value = float(root.evalf())
                if x_min <= value <= x_max:
                    roots.append(value)
    except Exception:
        pass
    return roots


# ---------------------------------------------------------------------------
# Sign-on-sides helper
# ---------------------------------------------------------------------------

def sign_on_sides(
    sym_expr: Expr,
    x: Symbol,
    x_value: float,
    x_min: float,
    x_max: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return the signs of sym_expr just to the left and right of x_value."""
    try:
        numeric_func = safe_lambdify(x, sym_expr)
    except Exception:
        return None, None

    span = x_max - x_min
    for scale in CFG.sign_steps:
        step   = max(span * scale, 1e-5)
        left_x = x_value - step
        right_x = x_value + step
        if left_x < x_min or right_x > x_max:
            continue
        try:
            left_y  = float(numeric_func(left_x))
            right_y = float(numeric_func(right_x))
            if _good_sign_value(left_y) and _good_sign_value(right_y):
                return float(np.sign(left_y)), float(np.sign(right_y))
        except Exception:
            continue
    return None, None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_critical_point(left_sign: Optional[float], right_sign: Optional[float]) -> str:
    if left_sign is None or right_sign is None:
        return "Điểm tới hạn"
    if left_sign > 0 and right_sign < 0:
        return "Cực đại địa phương"
    if left_sign < 0 and right_sign > 0:
        return "Cực tiểu địa phương"
    return "Điểm dừng phẳng"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_local_abs_minimum(abs_y: np.ndarray, i: int) -> bool:
    return (
        np.isfinite(abs_y[i - 1]) and np.isfinite(abs_y[i]) and np.isfinite(abs_y[i + 1])
        and abs_y[i] <= abs_y[i - 1]
        and abs_y[i] <= abs_y[i + 1]
        and abs_y[i] < CFG.tangent_threshold
    )


def _cluster_indexes(indexes: list[int], x_values: np.ndarray, tol: float) -> list[list[int]]:
    clusters: list[list[int]] = []
    for i in indexes:
        if not clusters or abs(x_values[i] - x_values[clusters[-1][-1]]) > tol:
            clusters.append([i])
        else:
            clusters[-1].append(i)
    return clusters


def _validate_root(numeric_func, x_value: float, tol: float) -> bool:
    try:
        return np.isfinite(float(numeric_func(x_value))) and abs(float(numeric_func(x_value))) <= tol
    except Exception:
        return False


def _good_sign_value(value: float) -> bool:
    return np.isfinite(value) and abs(value) > 1e-8


def _merge_tol(x_min: float, x_max: float) -> float:
    return max((x_max - x_min) * CFG.merge_tol_factor, 1e-3)


def _zero_tol(x_min: float, x_max: float) -> float:
    return max((x_max - x_min) * CFG.zero_snap_factor, 1e-3)


def _snap_zero(value: float, tol: float) -> float:
    return 0.0 if abs(value) < tol else float(value)


def _dedupe_values(values: list[float], x_min: float, x_max: float) -> list[float]:
    tol      = _merge_tol(x_min, x_max)
    zero_tol = _zero_tol(x_min, x_max)
    cleaned  = sorted(_snap_zero(v, zero_tol) for v in values if np.isfinite(v))
    groups: list[list[float]] = []
    for v in cleaned:
        if not groups or abs(v - groups[-1][-1]) > tol:
            groups.append([v])
        else:
            groups[-1].append(v)
    result: list[float] = []
    for group in groups:
        result.append(0.0 if any(abs(v) < zero_tol for v in group) else float(np.mean(group)))
    return result


def _dedupe_special_points(points: list[SpecialPoint], x_min: float, x_max: float) -> list[SpecialPoint]:
    tol      = _merge_tol(x_min, x_max)
    zero_tol = _zero_tol(x_min, x_max)
    priority = {
        "Cực đại địa phương": 1, "Cực tiểu địa phương": 1,
        "Điểm dừng phẳng": 2, "Điểm tới hạn": 3, "Điểm uốn": 4,
    }
    cleaned = [
        (_snap_zero(xv, zero_tol), yv, kind)
        for xv, yv, kind in points
        if np.isfinite(xv) and np.isfinite(yv)
    ]
    cleaned.sort(key=lambda item: (item[0], priority.get(item[2], 9)))
    final: list[SpecialPoint] = []
    for xv, yv, kind in cleaned:
        match_idx = _matching_point_index(final, xv, kind, tol)
        if match_idx is None:
            final.append((xv, yv, kind))
            continue
        old_x, old_y, old_kind = final[match_idx]
        new_x = 0.0 if abs(old_x) < zero_tol or abs(xv) < zero_tol else (old_x + xv) / 2
        final[match_idx] = (new_x, (old_y + yv) / 2, old_kind)
    final.sort(key=lambda item: item[0])
    return final


def _matching_point_index(points: list[SpecialPoint], xv: float, kind: str, tol: float) -> Optional[int]:
    for i, (old_x, _, old_kind) in enumerate(points):
        if kind == old_kind and abs(xv - old_x) <= tol:
            return i
    return None
