"""
experiments.py -- all numerical experiments for the MAT244 SINDy report.
Writes figures to figs/ and a machine-readable summary to results.json.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sindy as S

os.makedirs("figs", exist_ok=True)
RES = {}
RNG = lambda s: np.random.default_rng(s)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.4,
})
C = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d68910"]

LAMS = np.logspace(-4, 1.2, 60)


def sweep_lambda(Theta, Xdot, Xi_true, lams=LAMS):
    """Return per-lambda diagnostics + the 'recovery window' of lambdas that
    give exactly the true support."""
    ok, err, nact = [], [], []
    for lam in lams:
        Xi = S.stlsq(Theta, Xdot, lam)
        ok.append(S.support_match(Xi, Xi_true))
        err.append(S.coef_error(Xi, Xi_true))
        nact.append(S.n_active(Xi))
    ok = np.array(ok); err = np.array(err); nact = np.array(nact)
    win = lams[ok]
    return dict(ok=ok, err=err, nact=nact,
                recovered=bool(ok.any()),
                best_err=float(err[ok].min()) if ok.any() else float(err.min()),
                lam_lo=float(win.min()) if ok.any() else None,
                lam_hi=float(win.max()) if ok.any() else None,
                width=float(np.log10(win.max() / win.min())) if ok.any() else 0.0)


def prepare(sysm, X, method, dt):
    """Return (library input, derivative estimate) for a given method."""
    if method == "exact_deriv":
        return X, None
    if method == "fd":
        return X, S.deriv_fd(X, dt)
    if method == "savgol":
        from scipy.signal import savgol_filter
        m = X.shape[0]
        # window of fixed *duration* t_win, so smoothing is independent of dt
        w = int(round(sysm.t_win / dt))
        w = max(7, w + 1 - w % 2)
        w = min(w, m - (1 - m % 2))
        Xs = savgol_filter(X, w, 3, axis=0, mode="interp")
        return Xs, S.deriv_savgol(X, dt, window=w, order=3)
    raise ValueError(method)


# ======================================================================
# E0. Exact recovery on clean data (+ PySINDy cross-check)
# ======================================================================
def E0():
    out = {}
    for key, f in S.SYSTEMS.items():
        sysm = f()
        t, X, Xd = sysm.simulate()
        Theta, names = S.library(X, sysm.degree, sysm.var_names)
        Xi_true = sysm.true_Xi(names)
        rec = {}
        for method in ["exact_deriv", "fd", "savgol"]:
            Xl, Xdot = prepare(sysm, X, method, sysm.dt)
            Xdot = Xd if Xdot is None else Xdot
            Th, nm = S.library(Xl, sysm.degree, sysm.var_names)
            Xi = S.stlsq(Th, Xdot, 0.05)
            rec[method] = dict(support=bool(S.support_match(Xi, Xi_true)),
                               err=float(S.coef_error(Xi, Xi_true)),
                               model=S.print_model(Xi, nm, sysm.var_names))
        out[key] = dict(name=sysm.name, n_features=Theta.shape[1],
                        n_samples=X.shape[0], dt=sysm.dt, T=sysm.T,
                        cond=float(np.linalg.cond(Theta)), **rec)
        # coefficient table for the report (savgol on clean data)
        Xl, Xdot = prepare(sysm, X, "fd", sysm.dt)
        Th, nm = S.library(Xl, sysm.degree, sysm.var_names)
        Xi = S.stlsq(Th, Xdot, 0.05)
        out[key]["coefs"] = {nm[i]: [float(Xi[i, j]) for j in range(Xi.shape[1])]
                             for i in range(len(nm)) if np.abs(Xi[i]).max() > 0}
        out[key]["coefs_true"] = {nm[i]: [float(Xi_true[i, j]) for j in range(Xi.shape[1])]
                                  for i in range(len(nm)) if np.abs(Xi_true[i]).max() > 0}
    RES["E0"] = out

    # Figure 1: clean-data recovery, true vs identified trajectories
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for ax, (key, f) in zip(axes, S.SYSTEMS.items()):
        sysm = f()
        t, X, Xd = sysm.simulate()
        Th, nm = S.library(X, sysm.degree, sysm.var_names)
        Xi = S.stlsq(Th, S.deriv_fd(X, sysm.dt), 0.05)
        from scipy.integrate import solve_ivp
        def rhs(tt, u):
            th, _ = S.library(np.array(u)[None, :], sysm.degree, sysm.var_names)
            return (th @ Xi).ravel()
        sol = solve_ivp(rhs, (0, t[-1]), sysm.x0, t_eval=t, rtol=1e-9, atol=1e-11)
        Xr = sol.y.T
        if key == "lorenz":
            ax.plot(X[:, 0], X[:, 2], color=C[0], lw=0.7, label="true")
            ax.plot(Xr[:, 0], Xr[:, 2], "--", color=C[1], lw=0.7, label="identified")
            ax.set_xlabel("$x$"); ax.set_ylabel("$z$")
        else:
            ax.plot(X[:, 0], X[:, 1], color=C[0], lw=0.9, label="true")
            ax.plot(Xr[:, 0], Xr[:, 1], "--", color=C[1], lw=0.9, label="identified")
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        ax.set_title(sysm.name, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figs/fig1_clean.pdf"); plt.close(fig)


# ======================================================================
# E1. Numerical differentiation: truncation vs noise amplification
# ======================================================================
def E1():
    sysm = S.damped_oscillator()
    A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    dts = np.logspace(-3.3, -0.35, 26)
    sigs = [1e-4, 1e-3, 1e-2, 1e-1]
    rng = RNG(0)
    curves, opt_emp, opt_th, min_err = {}, [], [], []
    for sg in sigs:
        rmse = []
        for dt in dts:
            t, X, Xd = sysm.simulate(T=25.0, dt=dt)
            errs = []
            for trial in range(8):
                Xn = X + rng.normal(0, sg, X.shape)
                D = S.deriv_fd(Xn, dt)
                errs.append(np.sqrt(np.mean((D[2:-2] - Xd[2:-2]) ** 2)))
            rmse.append(np.mean(errs))
        rmse = np.array(rmse)
        curves[sg] = rmse
        opt_emp.append(dts[np.argmin(rmse)])
        min_err.append(rmse.min())
        # theory: h* = (3 sigma / M3)^(1/3), M3 = rms of third derivative
        t, X, _ = sysm.simulate(T=25.0, dt=0.01)
        X3 = (np.linalg.matrix_power(A, 3) @ X.T).T
        M3 = np.sqrt(np.mean(X3 ** 2))
        opt_th.append((3 * sg / M3) ** (1 / 3))
    RES["E1"] = dict(dts=dts.tolist(), sigmas=sigs, M3=float(M3),
                     rmse={str(k): v.tolist() for k, v in curves.items()},
                     h_opt_empirical=opt_emp, h_opt_theory=opt_th,
                     min_err=min_err,
                     slope_h=float(np.polyfit(np.log(sigs), np.log(opt_emp), 1)[0]),
                     slope_e=float(np.polyfit(np.log(sigs), np.log(min_err), 1)[0]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    for i, sg in enumerate(sigs):
        ax.loglog(dts, curves[sg], color=C[i], label=rf"$\sigma={sg:g}$")
        ax.axvline(opt_th[i], color=C[i], ls=":", lw=1.0)
    ax.set_xlabel(r"step size $h=\Delta t$"); ax.set_ylabel(r"RMSE of $\hat{\dot x}$")
    ax.set_title("Central differences: error vs. step size", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.loglog(sigs, opt_emp, "o", color=C[0], label="empirical $h^*$")
    ax.loglog(sigs, opt_th, "-", color=C[0], label=r"$(3\sigma/M_3)^{1/3}$")
    ax.loglog(sigs, min_err, "s", color=C[1], label="empirical min RMSE")
    ref = np.array(min_err)[0] * (np.array(sigs) / sigs[0]) ** (2 / 3)
    ax.loglog(sigs, ref, "--", color=C[1], label=r"$\propto\sigma^{2/3}$")
    ax.set_xlabel(r"noise level $\sigma$"); ax.set_ylabel("value")
    ax.set_title("Optimal step and attainable error", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figs/fig2_deriv.pdf"); plt.close(fig)


# ======================================================================
# E2. Noise robustness
# ======================================================================
def E2(n_trials=12):
    sigmas = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    out = {}
    for key in ["oscillator", "lorenz", "lotka"]:
        sysm = S.SYSTEMS[key]()
        t, X, Xd = sysm.simulate()
        _, nm = S.library(X, sysm.degree, sysm.var_names)
        Xi_true = sysm.true_Xi(nm)
        out[key] = {}
        for method in ["exact_deriv", "fd", "savgol"]:
            rate, errs = [], []
            for sg in sigmas:
                r, e = 0, []
                for k in range(n_trials if sg > 0 else 1):
                    rng = RNG(1000 * k + 7)
                    Xn = S.add_noise(X, sg, rng) if sg > 0 else X.copy()
                    Xl, Xdot = prepare(sysm, Xn, method, sysm.dt)
                    Xdot = Xd if Xdot is None else Xdot
                    Th, _ = S.library(Xl, sysm.degree, sysm.var_names)
                    d = sweep_lambda(Th, Xdot, Xi_true)
                    r += d["recovered"]; e.append(d["best_err"])
                n = n_trials if sg > 0 else 1
                rate.append(r / n); errs.append(float(np.median(e)))
            out[key][method] = dict(sigmas=sigmas, rate=rate, err=errs)
    RES["E2"] = out

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    labels = {"exact_deriv": r"exact $\dot X$ (noisy library only)",
              "fd": "finite differences", "savgol": "Savitzky--Golay"}
    for i, m in enumerate(["exact_deriv", "fd", "savgol"]):
        d = out["lorenz"][m]
        s = np.array(d["sigmas"]); s[0] = 3e-5
        axes[0].semilogx(s, d["rate"], "o-", color=C[i], label=labels[m], ms=3)
        axes[1].loglog(s, np.maximum(d["err"], 1e-16), "o-", color=C[i], ms=3)
    axes[0].set_xlabel(r"relative noise $\sigma_{\mathrm{rel}}$")
    axes[0].set_ylabel("exact-support recovery rate")
    axes[0].set_title("Lorenz: support recovery", fontsize=9)
    axes[0].legend(fontsize=7); axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel(r"relative noise $\sigma_{\mathrm{rel}}$")
    axes[1].set_ylabel("relative coefficient error")
    axes[1].set_title("Lorenz: coefficient accuracy", fontsize=9)
    fig.tight_layout(); fig.savefig("figs/fig3_noise.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for i, key in enumerate(["oscillator", "lorenz", "lotka"]):
        d = out[key]["savgol"]
        s = np.array(d["sigmas"]); s[0] = 3e-5
        ax.semilogx(s, d["rate"], "o-", color=C[i], ms=3,
                    label=S.SYSTEMS[key]().name)
    ax.set_xlabel(r"relative noise $\sigma_{\mathrm{rel}}$")
    ax.set_ylabel("recovery rate"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Savitzky--Golay pipeline", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figs/fig4_systems.pdf"); plt.close(fig)


# ======================================================================
# E3. Threshold sensitivity, the recovery window, and a predicted sigma_crit
# ======================================================================
def E3(n_trials=8):
    sysm = S.lorenz()
    t, X, Xd = sysm.simulate()
    _, nm = S.library(X, sysm.degree, sysm.var_names)
    Xi_true = sysm.true_Xi(nm)
    mask_true = np.abs(Xi_true) > 0
    xi_min = float(np.abs(Xi_true[mask_true]).min())

    # (a) illustrative lambda sweeps for four noise levels
    show = [0.0, 1e-3, 1e-2, 1e-1]
    curves = {}
    for sg in show:
        Xn = S.add_noise(X, sg, RNG(11)) if sg > 0 else X.copy()
        Xl, Xdot = prepare(sysm, Xn, "savgol", sysm.dt)
        Th, _ = S.library(Xl, sysm.degree, sysm.var_names)
        d = sweep_lambda(Th, Xdot, Xi_true)
        curves[sg] = dict(nact=d["nact"].tolist(), err=d["err"].tolist(),
                          lam_lo=d["lam_lo"], lam_hi=d["lam_hi"])

    # (b) window width vs noise (median over trials)
    sigmas = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    widths, rates = [], []
    for sg in sigmas:
        w, r = [], 0
        n = n_trials if sg > 0 else 1
        for k in range(n):
            Xn = S.add_noise(X, sg, RNG(23 * k + 4)) if sg > 0 else X.copy()
            Xl, Xdot = prepare(sysm, Xn, "savgol", sysm.dt)
            Th, _ = S.library(Xl, sysm.degree, sysm.var_names)
            d = sweep_lambda(Th, Xdot, Xi_true)
            w.append(d["width"]); r += d["recovered"]
        widths.append(float(np.median(w))); rates.append(r / n)

    # (c) growth of the spurious-coefficient scale -> predicted failure noise
    def window_edge(sysm, sig_list, seed0, n=6):
        t, X, Xd = sysm.simulate()
        _, nm = S.library(X, sysm.degree, sysm.var_names)
        Xt = sysm.true_Xi(nm)
        lo = []
        for sg in sig_list:
            v = []
            for k in range(n):
                Xn = S.add_noise(X, sg, RNG(seed0 + 31 * k))
                Xl, Xdot = prepare(sysm, Xn, "savgol", sysm.dt)
                Th, _ = S.library(Xl, sysm.degree, sysm.var_names)
                d = sweep_lambda(Th, Xdot, Xt)
                if d["recovered"]:
                    v.append(d["lam_lo"])
            lo.append(float(np.median(v)) if v else np.nan)
        return lo, float(np.abs(Xt[np.abs(Xt) > 0]).min())

    sig_fit = [1e-3, 3e-3, 1e-2, 3e-2]
    pert = {}
    for name, f, seed in [("oscillator", S.damped_oscillator, 100),
                          ("lorenz", S.lorenz, 200),
                          ("lotka", S.lotka_volterra, 300)]:
        e, xmin = window_edge(f(), sig_fit, seed)
        m = ~np.isnan(e)
        a_, b_ = np.polyfit(np.log10(np.array(sig_fit)[m]), np.log10(np.array(e)[m]), 1)
        pert[name] = dict(sigmas=sig_fit, err=e, xi_min=xmin, slope=float(a_),
                          sigma_pred=float(10 ** ((np.log10(xmin) - b_) / a_)))

    def observed_crit(rate, sig):
        sig = list(sig); rate = list(rate)
        for i in range(1, len(rate)):
            if rate[i] < 0.5 <= rate[i - 1]:
                s0, s1 = max(sig[i - 1], 1e-5), sig[i]
                r0, r1 = rate[i - 1], rate[i]
                f = (r0 - 0.5) / (r0 - r1)
                return float(10 ** (np.log10(s0) + f * (np.log10(s1) - np.log10(s0))))
        return None

    obs = {}
    if "E2" in RES:
        for k in ["oscillator", "lorenz", "lotka"]:
            d = RES["E2"][k]["savgol"]
            obs[k] = observed_crit(d["rate"], d["sigmas"])

    RES["E3"] = dict(lams=LAMS.tolist(), show=show,
                     curves={str(k): v for k, v in curves.items()},
                     sigmas=sigmas, widths=widths, rates=rates,
                     xi_min=xi_min, pert=pert, sigma_obs=obs,
                     n_true=int(mask_true.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.7))
    for i, sg in enumerate(show):
        c = curves[sg]
        axes[0].semilogx(LAMS, c["nact"], color=C[i],
                         label=rf"$\sigma_{{\rm rel}}={sg:g}$")
        if c["lam_lo"]:
            axes[0].axvspan(c["lam_lo"], c["lam_hi"], color=C[i], alpha=0.07)
        axes[1].loglog(LAMS, np.maximum(c["err"], 1e-16), color=C[i])
    axes[0].axhline(RES["E3"]["n_true"], color="k", ls="--", lw=0.9,
                    label="true no. of terms")
    axes[0].set_xlabel(r"threshold $\lambda$"); axes[0].set_ylabel("active terms")
    axes[0].set_title(r"Lorenz: sparsity vs. $\lambda$", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(r"threshold $\lambda$")
    axes[1].set_ylabel("relative coefficient error")
    axes[1].set_title(r"Accuracy vs. $\lambda$", fontsize=9)

    ax = axes[2]
    for i, (k, lab) in enumerate([("oscillator", "oscillator"),
                                  ("lorenz", "Lorenz"), ("lotka", "Lotka--Volterra")]):
        p = pert[k]
        ax.loglog(p["sigmas"], p["err"], "o", color=C[i], ms=3)
        xs = np.logspace(-3, 0.2, 50)
        a = p["slope"]; b = np.log10(p["err"][0]) - a * np.log10(p["sigmas"][0])
        ax.loglog(xs, 10 ** (a * np.log10(xs) + b), "-", color=C[i], lw=1.0, label=lab)
        ax.axhline(p["xi_min"], color=C[i], ls="--", lw=0.8)
        ax.plot([p["sigma_pred"]], [p["xi_min"]], "*", color=C[i], ms=9)
        if obs.get(k):
            ax.plot([obs[k]], [p["xi_min"]], "x", color=C[i], ms=6)
    ax.set_xlabel(r"relative noise $\sigma_{\rm rel}$")
    ax.set_ylabel(r"lower window edge $\lambda_{\rm lo}$")
    ax.set_title(r"Predicted vs. observed failure ($\star$ / $\times$)", fontsize=9)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig("figs/fig5_lambda.pdf"); plt.close(fig)


# ======================================================================
# E4. Data requirements: sampling rate and trajectory length
# ======================================================================
def E4(n_trials=5):
    sysm = S.lorenz()
    dts = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    Ts = [1.0, 2.0, 5.0, 10.0, 20.0]
    sg = 1e-3
    grid = np.zeros((len(Ts), len(dts)))
    for i, T in enumerate(Ts):
        for j, dt in enumerate(dts):
            t, X, Xd = sysm.simulate(T=T, dt=dt)
            _, nm = S.library(X, sysm.degree, sysm.var_names)
            Xi_true = sysm.true_Xi(nm)
            r = 0
            for k in range(n_trials):
                Xn = S.add_noise(X, sg, RNG(50 * k + 3))
                Xl, Xdot = prepare(sysm, Xn, "savgol", dt)
                Th, _ = S.library(Xl, sysm.degree, sysm.var_names)
                r += sweep_lambda(Th, Xdot, Xi_true)["recovered"]
            grid[i, j] = r / n_trials
    RES["E4"] = dict(dts=dts, Ts=Ts, sigma=sg, grid=grid.tolist())

    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(dts))); ax.set_xticklabels([f"{d:g}" for d in dts], fontsize=7)
    ax.set_yticks(range(len(Ts))); ax.set_yticklabels([f"{T:g}" for T in Ts], fontsize=7)
    ax.set_xlabel(r"sampling step $\Delta t$"); ax.set_ylabel("trajectory length $T$")
    ax.set_title(rf"Lorenz recovery rate, $\sigma_{{\rm rel}}={sg:g}$", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig("figs/fig6_data.pdf"); plt.close(fig)


# ======================================================================
# E5. STLSQ vs ordinary least squares
# ======================================================================
def E5():
    from scipy.integrate import solve_ivp
    sysm = S.lorenz()
    t, X, Xd = sysm.simulate()
    sg = 1e-3
    Xn = S.add_noise(X, sg, RNG(5))
    Xl, Xdot = prepare(sysm, Xn, "savgol", sysm.dt)
    Th, nm = S.library(Xl, sysm.degree, sysm.var_names)
    Xi_true = sysm.true_Xi(nm)
    Xi_ols = S.ols(Th, Xdot)
    d = sweep_lambda(Th, Xdot, Xi_true)
    lam = LAMS[np.argmin(np.where(d["ok"], d["err"], np.inf))]
    Xi_s = S.stlsq(Th, Xdot, lam)

    def sim(Xi, T=6.0):
        tt = np.arange(0, T, sysm.dt)
        def rhs(s, u):
            th, _ = S.library(np.array(u)[None, :], sysm.degree, sysm.var_names)
            return (th @ Xi).ravel()
        s = solve_ivp(rhs, (0, tt[-1]), sysm.x0, t_eval=tt, rtol=1e-9, atol=1e-11)
        return tt[:s.y.shape[1]], s.y.T

    t_t, X_t = sim(Xi_true); t_s, X_s = sim(Xi_s); t_o, X_o = sim(Xi_ols)
    RES["E5"] = dict(lam=float(lam), sigma=sg,
                     n_ols=S.n_active(Xi_ols), n_stlsq=S.n_active(Xi_s),
                     n_true=int((np.abs(Xi_true) > 0).sum()),
                     err_ols=float(S.coef_error(Xi_ols, Xi_true)),
                     err_stlsq=float(S.coef_error(Xi_s, Xi_true)),
                     res_ols=float(np.linalg.norm(Th @ Xi_ols - Xdot)),
                     res_stlsq=float(np.linalg.norm(Th @ Xi_s - Xdot)),
                     max_spurious_ols=float(np.abs(Xi_ols[np.abs(Xi_true) == 0]).max()),
                     model_stlsq=S.print_model(Xi_s, nm, sysm.var_names),
                     cond=float(np.linalg.cond(Th)))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    idx = np.arange(len(nm))
    axes[0].bar(idx - 0.2, np.abs(Xi_ols[:, 1]), 0.4, color=C[1], label="OLS")
    axes[0].bar(idx + 0.2, np.abs(Xi_s[:, 1]), 0.4, color=C[0], label="STLSQ")
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].set_xticks(idx); axes[0].set_xticklabels(nm, rotation=90, fontsize=5)
    axes[0].set_ylabel(r"$|\xi|$ in $\dot y$ equation")
    axes[0].axhline(lam, color="k", ls="--", lw=0.8, label=rf"$\lambda={lam:.2g}$")
    axes[0].set_title("Learned coefficients", fontsize=9); axes[0].legend(fontsize=7)
    axes[1].plot(t_t, X_t[:, 0], color="k", lw=0.9, label="true")
    axes[1].plot(t_s, X_s[:, 0], "--", color=C[0], lw=0.9, label="STLSQ")
    axes[1].plot(t_o, X_o[:, 0], ":", color=C[1], lw=1.0, label="OLS")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("$x(t)$")
    axes[1].set_title("Simulation of the learned models", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figs/fig7_ols.pdf"); plt.close(fig)


# ======================================================================
# E6. Library richness
# ======================================================================
def E6(n_trials=12):
    sysm = S.lorenz()
    t, X, Xd = sysm.simulate()
    degs = [2, 3, 4, 5]
    sigmas = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    conds, rates, nfeat = [], {}, []
    for dgi, dg in enumerate(degs):
        Th0, nm = S.library(X, dg, sysm.var_names)
        conds.append(float(np.linalg.cond(Th0)))
        nfeat.append(Th0.shape[1])
        Xi_true = sysm.true_Xi(nm)
        rr = []
        for sg in sigmas:
            r = 0
            n = n_trials if sg > 0 else 1
            for k in range(n):
                Xn = S.add_noise(X, sg, RNG(9 * k + 1)) if sg > 0 else X.copy()
                Xl, Xdot = prepare(sysm, Xn, "savgol", sysm.dt)
                Th, _ = S.library(Xl, dg, sysm.var_names)
                r += sweep_lambda(Th, Xdot, Xi_true)["recovered"]
            rr.append(r / n)
        rates[dg] = rr
    RES["E6"] = dict(degrees=degs, sigmas=sigmas, cond=conds,
                     n_features=nfeat, rates={str(k): v for k, v in rates.items()})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for i, dg in enumerate(degs):
        s = np.array(sigmas, dtype=float); s[0] = 3e-5
        axes[0].semilogx(s, rates[dg], "o-", color=C[i], ms=3,
                         label=f"degree {dg} ({nfeat[i]} terms)")
    axes[0].set_xlabel(r"relative noise $\sigma_{\rm rel}$")
    axes[0].set_ylabel("recovery rate"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Effect of library richness", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].semilogy(degs, conds, "o-", color=C[0])
    axes[1].set_xlabel("polynomial degree $p$")
    axes[1].set_ylabel(r"$\kappa_2(\Theta)$")
    axes[1].set_xticks(degs)
    axes[1].set_title("Conditioning of the library", fontsize=9)
    fig.tight_layout(); fig.savefig("figs/fig8_library.pdf"); plt.close(fig)


# ======================================================================
# E7. What actually controls the failure threshold?
# ======================================================================
def E7(n_trials=6):
    """For each system, measure the relative error injected into the two
    regression inputs (library and derivative) and the resulting equation
    error, and evaluate them at the empirically observed failure noise."""
    sigmas = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    out = {}
    for key, f in S.SYSTEMS.items():
        sysm = f()
        t, X, Xd = sysm.simulate()
        _, nm = S.library(X, sysm.degree, sysm.var_names)
        Xt = sysm.true_Xi(nm)
        Th0, _ = S.library(X, sysm.degree, sysm.var_names)
        rec = dict(sigmas=sigmas, eta_lib=[], eta_deriv=[], eta_eq=[],
                   xi_min=float(np.abs(Xt[np.abs(Xt) > 0]).min()),
                   xi_max=float(np.abs(Xt).max()),
                   cond=float(np.linalg.cond(Th0)))
        for sg in sigmas:
            a, b, c = [], [], []
            for k in range(n_trials):
                Xn = S.add_noise(X, sg, RNG(77 * k + 13))
                Xl, Xdot = prepare(sysm, Xn, "savgol", sysm.dt)
                Th, _ = S.library(Xl, sysm.degree, sysm.var_names)
                a.append(np.linalg.norm(Th - Th0) / np.linalg.norm(Th0))
                b.append(np.linalg.norm(Xdot - Xd) / np.linalg.norm(Xd))
                c.append(np.linalg.norm(Xdot - Th @ Xt) / np.linalg.norm(Xd))
            rec["eta_lib"].append(float(np.median(a)))
            rec["eta_deriv"].append(float(np.median(b)))
            rec["eta_eq"].append(float(np.median(c)))
        out[key] = rec
    # evaluate at observed failure noise (log-interpolation)
    if "E3" in RES and RES["E3"].get("sigma_obs"):
        for key, rec in out.items():
            sc = RES["E3"]["sigma_obs"].get(key)
            rec["sigma_obs"] = sc
            if sc:
                ls = np.log10(sigmas)
                for tag in ["eta_lib", "eta_deriv", "eta_eq"]:
                    rec[tag + "_at_crit"] = float(
                        10 ** np.interp(np.log10(sc), ls, np.log10(rec[tag])))
    RES["E7"] = out

    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for i, (key, lab) in enumerate([("oscillator", "oscillator"),
                                    ("lorenz", "Lorenz"),
                                    ("lotka", "Lotka--Volterra")]):
        r = out[key]
        ax.loglog(sigmas, r["eta_eq"], "o-", color=C[i], ms=3, label=lab)
        if r.get("sigma_obs"):
            ax.plot([r["sigma_obs"]], [r["eta_eq_at_crit"]], "*", color=C[i], ms=11)
    ax.set_xlabel(r"relative noise $\sigma_{\rm rel}$")
    ax.set_ylabel(r"equation error $\eta$")
    ax.set_title(r"$\eta$ at the observed failure point ($\star$)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figs/fig9_eta.pdf"); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    old = {}
    if os.path.exists("results.json"):
        old = json.load(open("results.json"))
    RES.update(old)
    for name in todo:
        t0 = time.time()
        globals()[name]()
        print(f"{name} done in {time.time()-t0:.1f}s", flush=True)
        json.dump(RES, open("results.json", "w"), indent=1)
