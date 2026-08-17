"""Exact polynomial propagation for the nilpotent phase family.

The supported circuit consists of translations ``exp(-i s.p)`` and
position-diagonal phase gates ``exp(-i V(x))`` with bounded-degree polynomial
``V``.  In the Heisenberg picture every such circuit has the form

    x -> x + c,        p -> p - grad W(x),

where ``W`` has the same bounded degree as the phase generators.  This module
stores ``W`` as a sparse multi-index polynomial and composes the circuit using
exact polynomial substitutions.  It also evaluates products of transformed
momenta on the multimode vacuum without a Fock cutoff.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

import numpy as np


def _clean(poly, atol=0.0):
    return {tuple(alpha): coeff for alpha, coeff in poly.items()
            if abs(coeff) > atol}


def constant(n, value=1.0):
    """Return the constant polynomial on ``n`` variables."""
    return {(0,) * n: complex(value)} if value != 0 else {}


def monomial(exponents, coefficient=1.0):
    """Return ``coefficient * x**exponents``."""
    return {tuple(exponents): complex(coefficient)} if coefficient != 0 else {}


def add(*polynomials):
    out = {}
    for poly in polynomials:
        for alpha, coefficient in poly.items():
            out[alpha] = out.get(alpha, 0.0j) + coefficient
    return _clean(out)


def scale(poly, factor):
    return _clean({alpha: factor * coefficient
                   for alpha, coefficient in poly.items()})


def derivative(poly, variable):
    """Differentiate a multi-index polynomial with respect to one variable."""
    out = {}
    for alpha, coefficient in poly.items():
        power = alpha[variable]
        if power == 0:
            continue
        beta = list(alpha)
        beta[variable] -= 1
        beta = tuple(beta)
        out[beta] = out.get(beta, 0.0j) + power * coefficient
    return _clean(out)


def multiply(left, right):
    """Multiply two commuting position polynomials."""
    out = {}
    for alpha, a in left.items():
        for beta, b in right.items():
            exponent = tuple(x + y for x, y in zip(alpha, beta))
            out[exponent] = out.get(exponent, 0.0j) + a * b
    return _clean(out)


def multiply_by_coordinate(poly, variable):
    out = {}
    for alpha, coefficient in poly.items():
        beta = list(alpha)
        beta[variable] += 1
        out[tuple(beta)] = coefficient
    return out


def shift(poly, displacement):
    """Substitute ``x -> x + displacement`` using the binomial theorem."""
    displacement = tuple(displacement)
    out = {}
    for alpha, coefficient in poly.items():
        terms = {(0,) * len(alpha): coefficient}
        for variable, power in enumerate(alpha):
            if power == 0:
                continue
            expanded = {}
            for beta, prefactor in terms.items():
                for retained_power in range(power + 1):
                    exponent = list(beta)
                    exponent[variable] += retained_power
                    factor = (comb(power, retained_power)
                              * displacement[variable] ** (power - retained_power))
                    exponent = tuple(exponent)
                    expanded[exponent] = expanded.get(exponent, 0.0j) + prefactor * factor
            terms = expanded
        for exponent, value in terms.items():
            out[exponent] = out.get(exponent, 0.0j) + value
    return _clean(out)


def evaluate(poly, coordinates):
    """Evaluate a polynomial on broadcast-compatible coordinate arrays."""
    out = 0.0
    for alpha, coefficient in poly.items():
        term = coefficient
        for coordinate, power in zip(coordinates, alpha):
            if power:
                term = term * coordinate ** power
        out = out + term
    return out


def effective_phase(n, gates):
    """Return the effective ``W`` for a chronologically ordered gate sequence.

    A phase gate is ``{"kind": "phase", "potential": V}`` and represents
    ``exp(-i V(x))``.  A translation is
    ``{"kind": "translation", "shift": s}`` and represents
    ``exp(-i s.p)``.  The returned polynomial satisfies
    ``U^dagger p_j U = p_j - partial_j W(x)``.
    """
    phase = {}
    for gate in reversed(gates):
        kind = gate["kind"]
        if kind == "phase":
            potential = gate["potential"]
            if any(len(alpha) != n for alpha in potential):
                raise ValueError("phase polynomial has the wrong number of variables")
            phase = add(phase, potential)
        elif kind == "translation":
            displacement = tuple(gate["shift"])
            if len(displacement) != n:
                raise ValueError("translation has the wrong number of components")
            phase = shift(phase, displacement)
        else:
            raise ValueError(f"unknown gate kind: {kind}")
    return phase


def _normal_moment(power, variance):
    if power % 2:
        return 0.0
    value = 1.0
    for k in range(power - 1, 0, -2):
        value *= k * variance
    return value


def vacuum_expectation(poly):
    """Evaluate a position polynomial in the vacuum, Var(x_j)=1/2."""
    total = 0.0j
    for alpha, coefficient in poly.items():
        moment = 1.0
        for power in alpha:
            moment *= _normal_moment(power, 0.5)
            if moment == 0:
                break
        total += coefficient * moment
    return total


def vacuum_momentum_moment(phase, indices):
    """Return ``<prod_j U^dagger p_indices[j] U>`` in the vacuum.

    Acting on ``f(x) psi_0(x)``, the transformed momentum is
    ``i (x_j f - partial_j f) - (partial_j W) f``.  Repeated application
    therefore reduces the product observable to a position polynomial whose
    vacuum overlap is evaluated analytically.
    """
    if phase:
        n = len(next(iter(phase)))
    elif indices:
        n = max(indices) + 1
    else:
        return 1.0 + 0.0j
    gradients = [derivative(phase, j) for j in range(n)]
    amplitude = constant(n)
    for variable in reversed(tuple(indices)):
        if variable < 0 or variable >= n:
            raise IndexError("momentum index outside the polynomial carrier")
        kinetic = scale(add(multiply_by_coordinate(amplitude, variable),
                            scale(derivative(amplitude, variable), -1.0)), 1.0j)
        shear = scale(multiply(gradients[variable], amplitude), -1.0)
        amplitude = add(kinetic, shear)
    return vacuum_expectation(amplitude)


def momentum_cumulant4(phase, variable=0):
    """Return the fourth cumulant of one transformed momentum."""
    moments = [1.0 + 0.0j]
    moments.extend(vacuum_momentum_moment(phase, (variable,) * order)
                   for order in range(1, 5))
    mean = moments[1]
    variance = moments[2] - mean ** 2
    central4 = (moments[4] - 4 * mean * moments[3]
                + 6 * mean ** 2 * moments[2] - 3 * mean ** 4)
    return central4 - 3 * variance ** 2


def connected_momentum_correlator(phase, first, second):
    mean_first = vacuum_momentum_moment(phase, (first,))
    mean_second = vacuum_momentum_moment(phase, (second,))
    second_moment = vacuum_momentum_moment(phase, (first, second))
    return second_moment - mean_first * mean_second


def homogeneous_polynomial(n, degree, coefficient):
    """Build a dense homogeneous polynomial from a coefficient callback."""
    out = {}
    for ordinal, modes in enumerate(combinations_with_replacement(range(n), degree)):
        alpha = [0] * n
        for mode in modes:
            alpha[mode] += 1
        value = coefficient(ordinal, modes)
        if value != 0:
            out[tuple(alpha)] = complex(value)
    return out


def algebra_dimension(n, degree):
    """Dimension of span{p_1,...,p_n} plus P_{<=degree}(x)."""
    return n + comb(n + degree, degree)


def translate_packed(poly, displacement):
    """Translate a bounded-degree multi-index polynomial.

    This is the degree-generic, symmetry-compressed counterpart of
    :func:`translate_cubic_tensors`.  The output represents ``poly(x+s)`` and
    stores each commuting monomial exactly once.
    """
    return shift(poly, displacement)


def effective_packed_phases(phases, shifts):
    """Compose translation--phase layers for any fixed polynomial degree."""
    if len(phases) != len(shifts):
        raise ValueError("each phase requires one preceding translation")
    if not phases:
        raise ValueError("at least one phase is required")
    result = {}
    for phase, displacement in reversed(list(zip(phases, shifts))):
        result = translate_packed(add(result, phase), displacement)
    return result


def translate_cubic_tensors(coefficients, displacement):
    """Translate a dense cubic polynomial in factorial-normalized form.

    ``coefficients=(c,l,Q,C)`` represents
    ``c + l_i x_i + Q_ij x_i x_j/2 + C_ijk x_i x_j x_k/6``, with symmetric
    ``Q`` and ``C``.  The returned tensors represent ``W(x+displacement)``.
    This dense representation makes fixed-degree scaling explicit: its largest
    object has ``n**3`` entries.
    """
    scalar, linear, quadratic, cubic = coefficients
    shift_vector = np.asarray(displacement)
    cubic_shift = np.einsum("ijk,k->ij", cubic, shift_vector, optimize=True)
    quadratic_shift = np.einsum("ij,j->i", quadratic, shift_vector, optimize=True)
    cubic_shift2 = np.einsum("ijk,j,k->i", cubic, shift_vector, shift_vector,
                             optimize=True)
    translated_scalar = (
        scalar
        + np.einsum("i,i->", linear, shift_vector, optimize=True)
        + 0.5 * np.einsum("ij,i,j->", quadratic, shift_vector, shift_vector,
                          optimize=True)
        + np.einsum("ijk,i,j,k->", cubic, shift_vector, shift_vector,
                    shift_vector, optimize=True) / 6.0
    )
    translated_linear = linear + quadratic_shift + 0.5 * cubic_shift2
    translated_quadratic = quadratic + cubic_shift
    return translated_scalar, translated_linear, translated_quadratic, cubic.copy()


def effective_cubic_tensors(cubic_phases, shifts):
    """Compose chronological translation--cubic-phase layers densely."""
    if len(cubic_phases) != len(shifts):
        raise ValueError("each cubic phase requires one preceding translation")
    if not cubic_phases:
        raise ValueError("at least one cubic phase is required")
    n = cubic_phases[0].shape[0]
    coefficients = (
        0.0,
        np.zeros(n, dtype=float),
        np.zeros((n, n), dtype=float),
        np.zeros((n, n, n), dtype=float),
    )
    for phase, displacement in reversed(list(zip(cubic_phases, shifts))):
        scalar, linear, quadratic, cubic = coefficients
        coefficients = (scalar, linear, quadratic, cubic + phase)
        coefficients = translate_cubic_tensors(coefficients, displacement)
    return coefficients


def evaluate_cubic_tensors(coefficients, coordinates):
    """Evaluate the dense factorial-normalized cubic representation."""
    scalar, linear, quadratic, cubic = coefficients
    x = np.asarray(coordinates)
    return (
        scalar
        + np.einsum("i,i->", linear, x, optimize=True)
        + 0.5 * np.einsum("ij,i,j->", quadratic, x, x, optimize=True)
        + np.einsum("ijk,i,j,k->", cubic, x, x, x, optimize=True) / 6.0
    )
