"""
sindy.py -- from-scratch implementation of Sparse Identification of Nonlinear
Dynamics (SINDy) with sequentially thresholded least squares (STLSQ).

MAT244 final project. No SINDy-specific library is used here; PySINDy is only
imported in validate.py as an independent cross-check.
"""
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter


# ----------------------------------------------------------------------
# 1. Candidate feature library  Theta(X)
# ----------------------------------------------------------------------
def poly_powers(n_states, degree):
    """All exponent tuples of total degree <= `degree` in `n_states` variables."""
    powers = []
    for d in range(degree + 1):
        for c in itertools.combinations_with_replacement(range(n_states), d):
            e = [0] * n_states
            for i in c:
                e[i] += 1
            powers.append(tuple(e))
    return powers


def poly_names(powers, var_names):
    names = []
    for e in powers:
        if sum(e) == 0:
            names.append("1")
            continue
        parts = []
        for v, p in zip(var_names, e):
            if p == 1:
                parts.append(v)
            elif p > 1:
                parts.append(f"{v}^{p}")
        names.append("".join(parts))
    return names


def library(X, degree=3, var_names=None):
    """Polynomial feature library. Returns (Theta, names)."""
    X = np.atleast_2d(X)
    m, n = X.shape
    if var_names is None:
        var_names = ["x", "y", "z", "w"][:n]
    powers = poly_powers(n, degree)
    Theta = np.empty((m, len(powers)))
    for j, e in enumerate(powers):
        col = np.ones(m)
        for i, p in enumerate(e):
            if p:
                col = col * X[:, i] ** p
        Theta[:, j] = col
    return Theta, poly_names(powers, var_names)


# ----------------------------------------------------------------------
# 2. Sequentially thresholded least squares
# ----------------------------------------------------------------------
def stlsq(Theta, Xdot, lam, max_iter=20, ridge=0.0):
    """
    Solve Xdot ~ Theta @ Xi, promoting sparsity by hard-thresholding at `lam`
    and re-solving least squares on the surviving terms until the support
    stops changing.
    """
    p = Theta.shape[1]
    n = Xdot.shape[1]

    def ls(A, b):
        if ridge > 0.0:
            A = np.vstack([A, np.sqrt(ridge) * np.eye(A.shape[1])])
            b = np.concatenate([b, np.zeros(A.shape[1])])
        return np.linalg.lstsq(A, b, rcond=None)[0]

    Xi = np.zeros((p, n))
    for j in range(n):
        Xi[:, j] = ls(Theta, Xdot[:, j])

    for _ in range(max_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        changed = False
        for j in range(n):
            big = ~small[:, j]
            if big.sum() == 0:
                continue
            new = np.zeros(p)
            new[big] = ls(Theta[:, big], Xdot[:, j])
            if not np.allclose(new, Xi[:, j]):
                changed = True
            Xi[:, j] = new
        if not changed:
            break
    return Xi


def ols(Theta, Xdot):
    return np.linalg.lstsq(Theta, Xdot, rcond=None)[0]


# ----------------------------------------------------------------------
# 3. Derivative estimation
# ----------------------------------------------------------------------
def deriv_fd(X, dt):
    """Second-order central differences (one-sided at the endpoints)."""
    D = np.empty_like(X)
    D[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    D[0] = (-3 * X[0] + 4 * X[1] - X[2]) / (2 * dt)
    D[-1] = (3 * X[-1] - 4 * X[-2] + X[-3]) / (2 * dt)
    return D


def deriv_savgol(X, dt, window=None, order=3):
    """Savitzky-Golay differentiation: local polynomial fit, differentiated."""
    m = X.shape[0]
    if window is None:
        window = max(order + 2, min(51, (m // 20) * 2 + 1))
    if window % 2 == 0:
        window += 1
    window = min(window, m - (1 - m % 2))
    return savgol_filter(X, window, order, deriv=1, delta=dt, axis=0, mode="interp")


DERIVS = {
    "exact": None,
    "fd": deriv_fd,
    "savgol": deriv_savgol,
}


# ----------------------------------------------------------------------
# 4. Test systems (data generators; their equations are the answer key)
# ----------------------------------------------------------------------
class System:
    def __init__(self, name, rhs, x0, degree, var_names, T, dt, true_terms,
                 t_win=0.5):
        self.name, self.rhs, self.x0 = name, rhs, x0
        self.degree, self.var_names = degree, var_names
        self.T, self.dt = T, dt
        self.t_win = t_win  # Savitzky-Golay window length, in time units
        self.true_terms = true_terms  # {(term_name, state_index): coefficient}

    def simulate(self, T=None, dt=None, x0=None, transient=0.0):
        T = self.T if T is None else T
        dt = self.dt if dt is None else dt
        x0 = self.x0 if x0 is None else x0
        t = np.arange(0, T + 1e-12, dt)
        sol = solve_ivp(self.rhs, (0, t[-1] + transient), x0,
                        t_eval=t + transient, rtol=1e-10, atol=1e-12,
                        dense_output=True)
        X = sol.y.T
        Xdot = np.array([self.rhs(0, xi) for xi in X])
        return t, X, Xdot

    def true_Xi(self, names):
        Xi = np.zeros((len(names), len(self.var_names)))
        for (term, j), c in self.true_terms.items():
            Xi[names.index(term), j] = c
        return Xi


def damped_oscillator():
    A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    return System(
        "Damped linear oscillator",
        lambda t, x: A @ x,
        np.array([2.0, 0.0]), 3, ["x", "y"], 25.0, 0.01,
        {("x", 0): -0.1, ("y", 0): 2.0, ("x", 1): -2.0, ("y", 1): -0.1},
    )


def lorenz():
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    def rhs(t, u):
        x, y, z = u
        return [s * (y - x), x * (r - z) - y, x * y - b * z]
    return System(
        "Lorenz system", rhs, np.array([-8.0, 7.0, 27.0]), 3, ["x", "y", "z"],
        10.0, 0.002,
        {("x", 0): -s, ("y", 0): s,
         ("x", 1): r, ("y", 1): -1.0, ("xz", 1): -1.0,
         ("z", 2): -b, ("xy", 2): 1.0},
        t_win=0.1,
    )


def lotka_volterra():
    a, b, c, d = 1.0, 0.1, 1.5, 0.075
    def rhs(t, u):
        x, y = u
        return [a * x - b * x * y, -c * y + d * x * y]
    return System(
        "Lotka-Volterra", rhs, np.array([10.0, 5.0]), 3, ["x", "y"], 30.0, 0.01,
        {("x", 0): a, ("xy", 0): -b, ("y", 1): -c, ("xy", 1): d},
    )


SYSTEMS = {"oscillator": damped_oscillator,
           "lorenz": lorenz,
           "lotka": lotka_volterra}


# ----------------------------------------------------------------------
# 5. Noise + scoring
# ----------------------------------------------------------------------
def add_noise(X, sigma_rel, rng):
    """Additive Gaussian noise with std sigma_rel * (std of each state)."""
    scale = X.std(axis=0)
    return X + rng.normal(0.0, sigma_rel * scale, size=X.shape)


def support_match(Xi, Xi_true, tol=0.0):
    return np.array_equal(np.abs(Xi) > tol, np.abs(Xi_true) > 0)


def coef_error(Xi, Xi_true):
    return np.linalg.norm(Xi - Xi_true) / np.linalg.norm(Xi_true)


def n_active(Xi):
    return int((np.abs(Xi) > 0).sum())


def print_model(Xi, names, var_names, tol=1e-12):
    lines = []
    for j, v in enumerate(var_names):
        terms = [f"{Xi[i, j]:+.4f} {names[i]}"
                 for i in range(len(names)) if abs(Xi[i, j]) > tol]
        lines.append(f"d{v}/dt = " + (" ".join(terms) if terms else "0"))
    return "\n".join(lines)


def fit(X, Xdot, degree, var_names, lam, ridge=0.0):
    Theta, names = library(X, degree, var_names)
    return stlsq(Theta, Xdot, lam, ridge=ridge), names, Theta
