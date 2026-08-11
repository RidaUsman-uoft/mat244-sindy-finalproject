"""validate.py -- independent cross-check of our from-scratch STLSQ against
PySINDy (de Silva et al., 2020). PySINDy is used ONLY here."""
import json
import numpy as np
import sindy as S
import pysindy as ps

out = {}
for key, f in S.SYSTEMS.items():
    sysm = f()
    t, X, Xd = sysm.simulate()
    Th, nm = S.library(X, sysm.degree, sysm.var_names)
    Xi_true = sysm.true_Xi(nm)
    ours = S.stlsq(Th, S.deriv_fd(X, sysm.dt), 0.05)

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.05, alpha=0.0),
        feature_library=ps.PolynomialLibrary(degree=sysm.degree),
        differentiation_method=ps.FiniteDifference(order=2),
    )
    model.fit(X, t=sysm.dt, feature_names=sysm.var_names)
    theirs = model.coefficients().T  # (features, states)
    # align feature ordering
    ref = [n.replace("^", "^") for n in model.get_feature_names()]
    def norm(s):
        return s.replace(" ", "").replace("1", "1") if s != "1" else "1"
    idx = []
    for name in nm:
        cand = [i for i, r in enumerate(ref) if norm(r) == norm(name)]
        idx.append(cand[0] if cand else None)
    aligned = np.zeros_like(ours)
    for i, j in enumerate(idx):
        if j is not None:
            aligned[i] = theirs[j]
    out[key] = dict(
        max_abs_diff=float(np.abs(ours - aligned).max()),
        same_support=bool(np.array_equal(np.abs(ours) > 0, np.abs(aligned) > 0)),
        our_err=float(S.coef_error(ours, Xi_true)),
        pysindy_err=float(S.coef_error(aligned, Xi_true)),
    )
    print(key, out[key])

json.dump(out, open("validation.json", "w"), indent=1)
