"""
Claim value model: given that a claim happens, how large is it?

Trained only on the shipments that actually produced a claim, because
claim_amount_rs is undefined for the rest - it is not zero, it simply does not
exist. The model therefore answers a conditional question, and it is multiplied
by the claim probability afterwards to give an expected loss.

The train/test split is *nested inside* the classifier's split: these models see
only claims that fall in the classifier's training set, and are scored on claims
in the classifier's test set. That costs a little accuracy compared with drawing
a fresh split over all claims, but it is what makes the cost analysis on slide 8
honest - there, the two models are combined over the classifier's test set, and
the value model must not have seen those rows.

Both models predict the claim amount directly in rupees. A log-transformed
target was tested first and fits the *shape* of the distribution better
(log-scale R2 around 0.37), but converting those predictions back to rupees -
with or without Duan's smearing correction - performed worse on exactly the
metrics this analysis is judged on (MAE, RMSE, R2 in rupees), because the
residual tail is too heavy for the back-transform to behave. Predicting rupees
directly is the honest choice here.

One ceiling is worth stating plainly: the size of a claim depends heavily on how
badly the goods were damaged, and damage severity is a post-event variable that
the leakage rule forbids. A moderate R2 is therefore the expected result, not a
modelling failure.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from .preprocessing import RANDOM_STATE, make_preprocessor


def build_regressors(numeric, categorical) -> dict:
    """Interpretable linear baseline plus a non-linear benchmark."""
    return {
        "linear": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", RandomForestRegressor(
                n_estimators=400, min_samples_leaf=8, max_features="sqrt",
                n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def predict_rupees(pipeline: Pipeline, X) -> np.ndarray:
    """Predicted claim amount, floored at zero (a claim cannot be negative)."""
    return np.clip(np.asarray(pipeline.predict(X), dtype=float), 0.0, None)


def regression_metrics(y_true_rs, y_pred_rs) -> dict:
    """MAE, RMSE and R-squared, all in rupees so they read as money."""
    y_true_rs = np.asarray(y_true_rs, dtype=float)
    y_pred_rs = np.asarray(y_pred_rs, dtype=float)
    resid = y_true_rs - y_pred_rs
    return {
        "mae": float(mean_absolute_error(y_true_rs, y_pred_rs)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(r2_score(y_true_rs, y_pred_rs)),
        "median_abs_error": float(np.median(np.abs(resid))),
        "mean_actual": float(np.mean(y_true_rs)),
        "median_actual": float(np.median(y_true_rs)),
        "mean_predicted": float(np.mean(y_pred_rs)),
        "n": int(len(y_true_rs)),
    }


def scatter_sample(y_true_rs, y_pred_rs, max_points: int = 900,
                   seed: int = RANDOM_STATE) -> dict:
    """A reproducible sample of the actual-vs-predicted cloud, with residuals."""
    y_true_rs = np.asarray(y_true_rs, dtype=float)
    y_pred_rs = np.asarray(y_pred_rs, dtype=float)
    n = len(y_true_rs)
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, max_points, replace=False)
        idx.sort()
    return {
        "actual": y_true_rs[idx].round(2).tolist(),
        "predicted": y_pred_rs[idx].round(2).tolist(),
        "residual": (y_true_rs[idx] - y_pred_rs[idx]).round(2).tolist(),
    }


def linear_drivers(pipeline: Pipeline, feature_names: list[str],
                   top_n: int = 10) -> list[dict]:
    """
    Largest coefficients of the linear claim-value model.

    Numeric predictors are standardised inside the pipeline, so a coefficient
    reads directly as "rupees added to the expected claim per one standard
    deviation increase"; one-hot columns read against the dropped level.
    """
    model = pipeline.named_steps["model"]
    coefs = np.asarray(model.coef_, dtype=float).ravel()
    rows = [{"feature": name, "coefficient": float(c)}
            for name, c in zip(feature_names, coefs)]
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return rows[:top_n]
