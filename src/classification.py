"""
Claim probability models: Logistic Regression, Random Forest and Bagging.

All three are trained on exactly the same pre-shipment feature matrix so the
comparison on the model-comparison page is fair. Nothing is calibrated or
re-weighted: the raw predicted probabilities are kept intact so that the
threshold slider on the simulator page moves a genuine operating point.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import RANDOM_STATE, encoded_feature_names, make_preprocessor


def build_models(numeric, categorical) -> dict:
    """The three classifiers, each behind its own preprocessing pipeline."""
    return {
        "logistic": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                                         random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=12,
                max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "bagging": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", BaggingClassifier(
                estimator=DecisionTreeClassifier(min_samples_leaf=25,
                                                 random_state=RANDOM_STATE),
                n_estimators=150, max_samples=0.8, max_features=0.8,
                n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Headline metrics at a fixed threshold, plus the threshold-free AUCs."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
    }


def confusion_at(y_true, proba, threshold: float) -> dict:
    """
    Confusion matrix and every derived rate at one threshold.

    This is the function behind the threshold slider - it is recomputed from the
    stored out-of-sample probability vector on every request, so the numbers on
    the screen are always the model's real behaviour at that cut-off.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": int(len(y_true)),
        "flagged": tp + fp,
        "flagged_pct": (tp + fp) / len(y_true) if len(y_true) else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def curve_points(y_true, proba, max_points: int = 300) -> dict:
    """ROC and PR curves, thinned to a size that draws smoothly in the browser."""
    fpr, tpr, roc_thr = roc_curve(y_true, proba)
    prec, rec, pr_thr = precision_recall_curve(y_true, proba)

    def thin(*arrays):
        n = len(arrays[0])
        if n <= max_points:
            idx = np.arange(n)
        else:
            idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        return [np.asarray(a)[idx].tolist() for a in arrays]

    roc_f, roc_t = thin(fpr, tpr)
    pr_p, pr_r = thin(prec, rec)
    return {
        "roc": {"fpr": roc_f, "tpr": roc_t},
        "pr": {"precision": pr_p, "recall": pr_r},
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
    }


def logistic_coefficients(pipeline: Pipeline, X_train, numeric, categorical,
                          top_n: int = 18) -> dict:
    """
    Coefficients, odds ratios and Wald p-values for the logistic model.

    Numeric predictors are standardised inside the pipeline, so an odds ratio is
    read as "multiplicative change in the odds of a claim per one standard
    deviation increase"; one-hot columns are read against the dropped level.
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    names = encoded_feature_names(prep, numeric, categorical)

    Z = prep.transform(X_train)
    Z = np.asarray(Z, dtype=float)
    beta = model.coef_.ravel()
    intercept = float(model.intercept_[0])

    # Wald standard errors from the observed information matrix.
    p = model.predict_proba(Z)[:, 1]
    W = np.clip(p * (1 - p), 1e-9, None)
    Zi = np.hstack([np.ones((Z.shape[0], 1)), Z])
    info = Zi.T * W @ Zi
    try:
        cov = np.linalg.pinv(info)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))[1:]
        z_stat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        pvals = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    except np.linalg.LinAlgError:            # pragma: no cover - defensive
        se = np.full_like(beta, np.nan)
        z_stat = np.full_like(beta, np.nan)
        pvals = np.full_like(beta, np.nan)

    rows = []
    for i, name in enumerate(names):
        rows.append({
            "feature": name,
            "coefficient": float(beta[i]),
            "odds_ratio": float(np.exp(beta[i])),
            "std_error": float(se[i]),
            "z": float(z_stat[i]),
            "p_value": float(pvals[i]),
            "significant": bool(pvals[i] < 0.05),
        })
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return {
        "intercept": intercept,
        "n_features": len(names),
        "n_significant": int(sum(r["significant"] for r in rows)),
        "top": rows[:top_n],
        "all": rows,
    }


def grouped_importance(pipeline: Pipeline, numeric, categorical,
                       top_n: int = 12) -> list[dict]:
    """
    Random Forest / Bagging importance summed back to the original variables.

    A one-hot encoded variable is spread over many columns; summing the parts
    gives an importance that a business reader can act on ("packaging quality"
    rather than "packaging_type=Shrink wrap").
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    names = encoded_feature_names(prep, numeric, categorical)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:                                     # BaggingClassifier of trees
        imp = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)

    totals: dict[str, float] = {}
    for name, value in zip(names, imp):
        base = name.split("=", 1)[0]
        totals[base] = totals.get(base, 0.0) + float(value)

    total = sum(totals.values()) or 1.0
    rows = [{"variable": k, "importance": v / total} for k, v in totals.items()]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows[:top_n]


def threshold_sweep(y_true, proba, step: float = 0.01) -> dict:
    """
    Every metric at every threshold, pre-computed once.

    Page 7 plots precision, recall and F1 as continuous functions of the
    threshold. Sweeping here rather than in the browser keeps that curve exact
    and keeps the slider instant.
    """
    thresholds = np.round(np.arange(step, 1.0, step), 4)
    out = {"threshold": thresholds.tolist(), "precision": [], "recall": [],
           "f1": [], "specificity": [], "accuracy": [], "flagged_pct": []}
    for t in thresholds:
        row = confusion_at(y_true, proba, float(t))
        for key in ("precision", "recall", "f1", "specificity", "accuracy", "flagged_pct"):
            out[key].append(row[key])
        # Precision is undefined once nothing is flagged at all; recording it as
        # None breaks the line on the chart instead of drawing a false drop to 0.
        if row["tp"] + row["fp"] == 0:
            out["precision"][-1] = None
    best = int(np.argmax(out["f1"]))
    out["best_f1_threshold"] = float(thresholds[best])
    out["best_f1"] = float(out["f1"][best])
    return out


def probability_histogram(proba, bins: int = 40) -> dict:
    """Distribution of predicted probabilities on the test set."""
    counts, edges = np.histogram(np.asarray(proba), bins=bins, range=(0.0, 1.0))
    return {"counts": counts.astype(int).tolist(),
            "edges": np.round(edges, 4).tolist()}
