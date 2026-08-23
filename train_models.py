"""
Offline training run for the freight damage claims presentation.

Run this once:

    python train_models.py

It trains every model, evaluates them on a held-out test set, pre-computes the
figures the presentation needs, and writes everything to ``models/``. The Flask
app only ever loads those artefacts, so starting the server is instant and the
numbers on screen never drift between runs.
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import classification as cls
from src import clustering as clu
from src import regression as reg
from src.preprocessing import (DROPPED_WITH_REASON, FEATURE_GROUPS, MODELS_DIR,
                               RANDOM_STATE, TARGET_CLS, TARGET_REG, TEST_SIZE,
                               baseline_row, carrier_profiles, dictionary_split,
                               encoded_feature_names, feature_columns, load_raw,
                               pretty)

# Categorical / numeric variables offered in the exploratory analysis dropdown.
EDA_CATEGORICALS = ["carrier_type", "transport_mode", "product_category",
                    "packaging_type", "vehicle_type", "shipment_priority",
                    "season", "origin_zone", "destination_zone"]
EDA_NUMERICS = ["fragility_class", "packaging_quality_score", "distance_km",
                "vehicle_utilization_pct", "carrier_historical_damage_rate",
                "route_complexity_score", "declared_value_rs", "vehicle_age_years",
                "historical_lane_claim_rate", "loading_quality_score"]


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Exploratory analysis, pre-computed so page 3 is instant
# ----------------------------------------------------------------------------
def build_eda(df: pd.DataFrame) -> dict:
    n = len(df)
    claims = df[df[TARGET_CLS] == 1]
    base_rate = float(df[TARGET_CLS].mean())

    by_category = {}
    for col in EDA_CATEGORICALS:
        if col not in df.columns:
            continue
        g = df.groupby(col).agg(n=(TARGET_CLS, "size"), rate=(TARGET_CLS, "mean"))
        g = g[g["n"] >= 100].sort_values("rate")
        amounts = claims.groupby(col)[TARGET_REG].mean().reindex(g.index)
        top, bottom = g.index[-1], g.index[0]
        by_category[col] = {
            "label": pretty(col),
            "levels": [str(x) for x in g.index],
            "rate": g["rate"].round(5).tolist(),
            "count": g["n"].astype(int).tolist(),
            "avg_claim_value": [None if pd.isna(v) else float(round(v, 2))
                                for v in amounts],
            "observations": [
                f"{top} carries the highest claim rate at {g['rate'].iloc[-1]*100:.1f}%, "
                f"against {g['rate'].iloc[0]*100:.1f}% for {bottom} - a spread of "
                f"{(g['rate'].iloc[-1] - g['rate'].iloc[0])*100:.1f} percentage points.",
                f"The portfolio average is {base_rate*100:.1f}%, so {top} runs "
                f"{g['rate'].iloc[-1]/base_rate:.2f}x the book average.",
            ],
        }

    by_numeric = {}
    for col in EDA_NUMERICS:
        if col not in df.columns:
            continue
        s = df[col]
        # A discrete variable (fragility 1-5, packaging quality 1-5) must be
        # grouped on its own values; quantile binning would silently merge them.
        if s.nunique() <= 10:
            bins = s
        else:
            try:
                bins = pd.qcut(s, q=10, duplicates="drop")
            except ValueError:                # pragma: no cover - defensive
                bins = pd.cut(s, bins=min(10, max(2, s.nunique())))
        g = df.groupby(bins, observed=True).agg(
            n=(TARGET_CLS, "size"), rate=(TARGET_CLS, "mean"), mid=(col, "median"))
        g = g[g["n"] >= 50]
        rates = g["rate"].to_numpy()
        direction = "rises" if rates[-1] > rates[0] else "falls"
        by_numeric[col] = {
            "label": pretty(col),
            "bin_labels": [str(i) for i in g.index.astype(str)],
            "mid": g["mid"].round(4).tolist(),
            "rate": g["rate"].round(5).tolist(),
            "count": g["n"].astype(int).tolist(),
            "correlation": float(df[col].corr(df[TARGET_CLS])),
            "observations": [
                f"Claim rate {direction} from {rates[0]*100:.1f}% in the lowest band to "
                f"{rates[-1]*100:.1f}% in the highest band of {pretty(col).lower()}.",
                f"Point-biserial correlation with the claim flag is "
                f"{df[col].corr(df[TARGET_CLS]):+.3f}.",
            ],
        }

    amounts = claims[TARGET_REG].to_numpy(dtype=float)
    hist_counts, hist_edges = np.histogram(np.log10(np.clip(amounts, 1, None)), bins=30)

    return {
        "n_shipments": int(n),
        "n_claims": int(len(claims)),
        "claim_rate": base_rate,
        "target_counts": {"no_claim": int((df[TARGET_CLS] == 0).sum()),
                          "claim": int(len(claims))},
        "date_range": [str(df["shipment_date"].min().date()),
                       str(df["shipment_date"].max().date())],
        "n_lanes": int(df["lane_id"].nunique()),
        "n_carriers": int(df["carrier_id"].nunique()),
        "total_claim_value": float(amounts.sum()),
        "claim_amount_stats": {
            "min": float(amounts.min()), "p25": float(np.percentile(amounts, 25)),
            "median": float(np.median(amounts)), "mean": float(amounts.mean()),
            "p75": float(np.percentile(amounts, 75)),
            "p95": float(np.percentile(amounts, 95)), "max": float(amounts.max()),
        },
        "claim_amount_hist": {
            "log10_edges": hist_edges.round(4).tolist(),
            "counts": hist_counts.astype(int).tolist(),
        },
        "monthly": (df.assign(m=df["shipment_date"].dt.to_period("M").astype(str))
                      .groupby("m").agg(n=(TARGET_CLS, "size"), rate=(TARGET_CLS, "mean"))
                      .reset_index().to_dict(orient="list")),
        "by_category": by_category,
        "by_numeric": by_numeric,
        "default_category": "carrier_type",
        "default_numeric": "fragility_class",
    }


# ----------------------------------------------------------------------------
# Leakage table for page 2
# ----------------------------------------------------------------------------
def build_leakage(split: dict, numeric: list[str], categorical: list[str]) -> dict:
    used = set(numeric + categorical)
    return {
        "used": sorted(used),
        "n_used": len(used),
        "pre_shipment": [
            {"name": c, "label": pretty(c), "used": c in used,
             "reason": DROPPED_WITH_REASON.get(c, ""),
             "description": split["descriptions"].get(c, "")}
            for c in split["pre_shipment"]
        ],
        "in_transit": [
            {"name": c, "label": pretty(c),
             "description": split["descriptions"].get(c, "")}
            for c in split["in_transit"]
        ],
        "post_event": [
            {"name": c, "label": pretty(c),
             "description": split["descriptions"].get(c, "")}
            for c in split["post_event"]
        ],
        "targets": split["targets"],
        "rule": ("Only information available at the moment the shipment decision is "
                 "made may be used to predict claim risk. Variables observed during "
                 "transit and variables that exist only because damage already "
                 "happened are excluded from every claim-probability model."),
    }


def main() -> None:
    started = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    log("loading data")
    df = load_raw()
    split = dictionary_split()
    numeric, categorical = feature_columns(df)
    features = numeric + categorical
    log(f"{len(df):,} shipments | {len(features)} model features "
        f"({len(numeric)} numeric, {len(categorical)} categorical)")
    log(f"excluded as leakage: {len(split['in_transit'])} in-transit + "
        f"{len(split['post_event'])} post-event columns")

    # ---------------- classification -------------------------------------
    X = df[features]
    y = df[TARGET_CLS].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    log(f"train {len(X_train):,} / test {len(X_test):,} "
        f"(claim rate {y_train.mean():.3f} / {y_test.mean():.3f})")

    models = cls.build_models(numeric, categorical)
    probs, model_meta = {}, {}
    display_names = {"logistic": "Logistic Regression",
                     "random_forest": "Random Forest",
                     "bagging": "Bagging (Decision Trees)"}
    for key, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        p_test = pipe.predict_proba(X_test)[:, 1]
        probs[key] = p_test
        metrics = cls.classification_metrics(y_test, p_test, 0.5)
        curves = cls.curve_points(y_test, p_test)
        model_meta[key] = {
            "key": key,
            "name": display_names[key],
            "metrics": metrics,
            "curves": curves,
            "train_seconds": round(time.time() - t0, 2),
            "prob_min": float(p_test.min()),
            "prob_max": float(p_test.max()),
            "prob_mean": float(p_test.mean()),
            "sweep": cls.threshold_sweep(y_test, p_test),
            "prob_hist": cls.probability_histogram(p_test),
        }
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"clf_{key}.joblib"))
        log(f"  {display_names[key]:<26} ROC-AUC {metrics['roc_auc']:.4f} | "
            f"PR-AUC {metrics['pr_auc']:.4f} | {time.time()-t0:.1f}s")

    log("logistic coefficients + Wald p-values")
    coefs = cls.logistic_coefficients(models["logistic"], X_train, numeric, categorical)
    log(f"  {coefs['n_significant']}/{coefs['n_features']} coefficients significant at 5%")

    importances = {
        "random_forest": cls.grouped_importance(models["random_forest"], numeric, categorical),
        "bagging": cls.grouped_importance(models["bagging"], numeric, categorical),
    }

    np.savez_compressed(
        os.path.join(MODELS_DIR, "eval_probs.npz"),
        y_test=y_test.to_numpy(),
        **{f"p_{k}": v for k, v in probs.items()},
    )

    # ---------------- claim value regression ------------------------------
    log("claim value regression (claims only)")
    claims = df[df[TARGET_CLS] == 1].copy()
    Xr = claims[features]
    yr = claims[TARGET_REG].astype(float)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    regressors = reg.build_regressors(numeric, categorical)
    reg_meta = {}
    reg_display = {"linear": "Linear Regression", "random_forest": "Random Forest Regressor"}
    for key, pipe in regressors.items():
        pipe.fit(Xr_tr, yr_tr)
        pred_rs = reg.predict_rupees(pipe, Xr_te)
        metrics = reg.regression_metrics(yr_te, pred_rs)
        reg_meta[key] = {
            "key": key,
            "name": reg_display[key],
            "metrics": metrics,
            "scatter": reg.scatter_sample(yr_te, pred_rs),
        }
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"reg_{key}.joblib"))
        log(f"  {reg_display[key]:<26} MAE Rs {metrics['mae']:,.0f} | "
            f"RMSE Rs {metrics['rmse']:,.0f} | R2 {metrics['r2']:.3f}")

    enc_names = encoded_feature_names(
        regressors["linear"].named_steps["prep"], numeric, categorical)
    reg_meta["linear"]["drivers"] = reg.linear_drivers(regressors["linear"], enc_names)
    best_reg = max(reg_meta, key=lambda k: reg_meta[k]["metrics"]["r2"])

    # ---------------- lane clustering -------------------------------------
    log("K-Means lane archetypes")
    lanes = clu.build_lane_table(df)
    clusters = clu.cluster_lanes(lanes)
    log(f"  {len(lanes)} lanes clustered | elbow K={clusters['elbow_k']}, best silhouette K={clusters['best_k_silhouette']}")

    # ---------------- everything else the pages need ----------------------
    baseline = baseline_row(df, numeric, categorical)
    control_meta = []
    for col in numeric:
        s = df[col]
        lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
        binary = bool(set(s.dropna().unique()) <= {0, 1})
        # Money and weight span several orders of magnitude, so a linear slider
        # would park the median in the first few percent of its travel. Those get
        # a log-scaled slider instead; everything else stays linear.
        log_scale = bool((not binary) and lo > 0 and float(s.skew()) > 1.5)
        control_meta.append({
            "name": col, "label": pretty(col), "kind": "numeric",
            "min": float(np.round(lo, 4)),
            "max": float(np.round(hi, 4)),
            "median": float(np.round(s.median(), 4)),
            "step": float(np.round(max((hi - lo) / 100, 0.01), 4)),
            "is_binary": binary,
            "scale": "log" if log_scale else "linear",
            "skew": float(np.round(s.skew(), 3)),
        })
    for col in categorical:
        control_meta.append({
            "name": col, "label": pretty(col), "kind": "categorical",
            "levels": sorted(str(v) for v in df[col].dropna().unique()),
        })

    best_cls = max(model_meta, key=lambda k: model_meta[k]["metrics"]["pr_auc"])
    artefacts = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "rows": int(len(df)), "columns": int(df.shape[1]),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "claim_rows": int(len(claims)),
            "reg_train_rows": int(len(Xr_tr)), "reg_test_rows": int(len(Xr_te)),
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
            "missing_pct": {c: float(round(df[c].isna().mean() * 100, 2))
                            for c in features if df[c].isna().any()},
        },
        "features": {"numeric": numeric, "categorical": categorical,
                     "groups": FEATURE_GROUPS,
                     "labels": {c: pretty(c) for c in features}},
        "leakage": build_leakage(split, numeric, categorical),
        "classification": model_meta,
        "best_classifier": best_cls,
        "coefficients": coefs,
        "importances": importances,
        "regression": reg_meta,
        "best_regressor": best_reg,
        "clusters": clusters,
        "eda": build_eda(df),
        "baseline": baseline,
        "controls": control_meta,
        "carriers": carrier_profiles(df),
    }

    out = os.path.join(MODELS_DIR, "artifacts.json")
    with open(out, "w") as fh:
        json.dump(artefacts, fh, separators=(",", ":"), allow_nan=False, default=float)
    size_mb = os.path.getsize(out) / 1e6
    log(f"wrote {out} ({size_mb:.2f} MB)")
    log(f"best classifier by PR-AUC: {display_names[best_cls]}")
    log(f"best claim-value model by R2: {reg_display[best_reg]}")
    log(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
