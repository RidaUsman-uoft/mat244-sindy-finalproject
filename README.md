# mat244-sindy-finalproject
# Data-driven Discovery of Nonlinear Dynamical Systems via Sparse Regression (SINDy)

Code for the MAT244 final project. Implements SINDy with sequentially
thresholded least squares (STLSQ) from scratch, and studies when it works
and when it fails.

- `sindy.py` — polynomial library, STLSQ, derivative estimators, test systems
- `experiments.py` — all numerical experiments; writes figures and `results.json`
- `validate.py` — cross-check of the from-scratch STLSQ against PySINDy
- `report_figure.py` — assembles Figure 1 of the report from `results.json`

## Running

    pip install numpy scipy matplotlib pysindy
    python experiments.py     # all experiments, figures to figs/
    python validate.py        # PySINDy cross-check
    python report_figure.py # Figure 1 of the report -> figs/figMain.pdf

PySINDy is used only for validation, never to produce results.
