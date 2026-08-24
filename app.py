"""
Freight Damage Claims Analytics - interactive presentation server.

    python train_models.py     # once, builds models/
    python app.py              # then open http://127.0.0.1:5000

Every number this server returns is computed from the trained models or from the
held-out evaluation set. Nothing is hardcoded, sampled from a template, or
invented: if a value cannot be computed the endpoint says so rather than
guessing.
"""

from __future__ import annotations

import json
import os
import shutil

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from src import cost_benefit as cb
from src.classification import confusion_at
from src.preprocessing import (BASE_DIR, MODELS_DIR, TARGET_CLS, load_raw,
                               to_frame)

app = Flask(__name__)
# Flask's JSON provider alphabetises keys by default, which would reorder the
# business variable groups and the model list on screen.
app.json.sort_keys = False
# Slide wording lives in the template; reloading it on change means editing the
# deck does not require restarting the server mid-preparation.
app.config["TEMPLATES_AUTO_RELOAD"] = True

ARTIFACTS_PATH = os.path.join(MODELS_DIR, "artifacts.json")
PROBS_PATH = os.path.join(MODELS_DIR, "eval_probs.npz")
CLASSIFIER_KEYS = ("logistic", "random_forest", "bagging")


def _load_for_inference(path: str):
    """
    Load a trained pipeline and tune it for single-row scoring.

    The ensembles were trained with n_jobs=-1, which is right for fitting on
    18,750 rows and badly wrong for predicting on one: joblib's thread pool costs
    far more than the trees it parallelises, turning a 5 ms call into 300 ms.
    Every slider on the deck goes through this path, so it has to be fast.
    """
    pipeline = joblib.load(path)
    model = pipeline.named_steps.get("model")
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return pipeline


class ModelStore:
    """Loads the training artefacts once and serves them to every request."""

    def __init__(self) -> None:
        if not os.path.exists(ARTIFACTS_PATH):
            raise SystemExit(
                "models/artifacts.json not found - run `python train_models.py` first.")

        with open(ARTIFACTS_PATH) as fh:
            self.art = json.load(fh)

        self.numeric = self.art["features"]["numeric"]
        self.categorical = self.art["features"]["categorical"]
        self.baseline = self.art["baseline"]

        self.classifiers = {
            key: _load_for_inference(os.path.join(MODELS_DIR, f"clf_{key}.joblib"))
            for key in CLASSIFIER_KEYS
        }
        self.regressors = {
            key: _load_for_inference(os.path.join(MODELS_DIR, f"reg_{key}.joblib"))
            for key in ("linear", "random_forest")
        }

        blob = np.load(PROBS_PATH)
        self.y_test = blob["y_test"]
        self.eval_probs = {k: blob[f"p_{k}"] for k in CLASSIFIER_KEYS}
        # What each held-out shipment actually cost, and what the value model
        # would have quoted for it. The cost analysis backtests against these.
        self.claim_amount_test = blob["claim_amount_test"]
        self.pred_value_test = blob["pred_value_test"]
        self.mean_claim = float(self.claim_amount_test[self.y_test == 1].mean())

        self.best_classifier = self.art["best_classifier"]
        self.best_regressor = self.art["best_regressor"]

        # Lane -> archetype lookup, rebuilt per K so page 4 and page 10 agree.
        lanes = self.art["clusters"]["lanes"]
        self.lane_index = {row["lane_id"]: i for i, row in enumerate(lanes)}
        self.lane_rows = lanes

        # Claim-amount distribution of the observed claims, used to place a
        # single prediction in context ("this is a large claim for this book").
        self.claim_stats = self.art["eda"]["claim_amount_stats"]

    # -- prediction ---------------------------------------------------------
    def expected_loss(self, payload: dict) -> dict:
        """
        Expected loss from the two primary models only.

        Used by the lever search, which scores several hypothetical shipments per
        request; running all five models each time would triple the work for
        numbers the caller never reads.
        """
        frame = to_frame(payload, self.baseline, self.numeric, self.categorical)
        p = float(self.classifiers[self.best_classifier].predict_proba(frame)[0, 1])
        v = float(max(self.regressors[self.best_regressor].predict(frame)[0], 0.0))
        return {"probability": p, "claim_value": v, "expected_loss": p * v}

    def predict(self, payload: dict) -> dict:
        """Claim probability, conditional claim value and expected loss."""
        frame = to_frame(payload, self.baseline, self.numeric, self.categorical)

        probabilities = {
            key: float(pipe.predict_proba(frame)[0, 1])
            for key, pipe in self.classifiers.items()
        }
        claim_values = {
            key: float(max(pipe.predict(frame)[0], 0.0))
            for key, pipe in self.regressors.items()
        }

        p_primary = probabilities[self.best_classifier]
        v_primary = claim_values[self.best_regressor]
        return {
            "probabilities": probabilities,
            "claim_values": claim_values,
            "primary": {
                "classifier": self.best_classifier,
                "classifier_name": self.art["classification"][self.best_classifier]["name"],
                "regressor": self.best_regressor,
                "regressor_name": self.art["regression"][self.best_regressor]["name"],
                "probability": p_primary,
                "claim_value": v_primary,
                "expected_loss": p_primary * v_primary,
            },
            "percentiles": {
                key: float((self.eval_probs[key] < p).mean())
                for key, p in probabilities.items()
            },
            "claim_value_percentile": self._claim_value_percentile(v_primary),
            "inputs": {k: (float(v) if k in self.numeric else v)
                       for k, v in frame.iloc[0].items()},
        }

    # Levers an operations team can actually pull, with the setting each would
    # move to. Used to answer "what is the single best thing to change here?"
    LEVERS = [
        ("packaging_quality_score", 5.0, "upgrade packaging to quality 5"),
        ("palletized", 1.0, "palletise the shipment"),
        ("vehicle_utilization_pct", 0.75, "load the vehicle to 75% instead"),
        ("route_complexity_score", 2.5, "route via a simpler lane (complexity 2.5)"),
        ("loading_quality_score", 5.0, "secure the cargo to loading quality 5"),
    ]

    def best_lever(self, shipment: dict, baseline_loss: float) -> dict | None:
        """
        Score each lever and return the one that lowers expected loss the most.

        Every candidate is a real forward pass through the same trained models,
        so the answer changes with the shipment rather than being a fixed list.
        """
        best = None
        for name, target, phrasing in self.LEVERS:
            current = shipment.get(name, self.baseline.get(name))
            if current is None or abs(float(current) - target) < 1e-9:
                continue
            trial = dict(shipment)
            trial[name] = target
            result = self.expected_loss(trial)
            saving = baseline_loss - result["expected_loss"]
            if best is None or saving > best["saving"]:
                best = {
                    "variable": name, "action": phrasing,
                    "from": float(current), "to": target,
                    "saving": saving,
                    "new_probability": result["probability"],
                    "new_expected_loss": result["expected_loss"],
                }
        return best

    def _claim_value_percentile(self, value: float) -> float:
        """Roughly where a predicted claim sits in the observed claim book."""
        marks = [("min", 0.0), ("p25", 0.25), ("median", 0.5),
                 ("p75", 0.75), ("p95", 0.95), ("max", 1.0)]
        prev_v, prev_q = self.claim_stats["min"], 0.0
        for name, q in marks[1:]:
            cur = self.claim_stats[name]
            if value <= cur:
                span = cur - prev_v
                frac = 0.0 if span <= 0 else (value - prev_v) / span
                return float(prev_q + frac * (q - prev_q))
            prev_v, prev_q = cur, q
        return 1.0

    def risk_band(self, probability: float, threshold: float) -> str:
        """
        Low / Medium / High relative to the operating threshold in force.

        The threshold is the point at which the business decides to act, so
        "High" means at or above it, and "Medium" means close enough to it to be
        worth a second look.
        """
        if probability >= threshold:
            return "High"
        if probability >= 0.6 * threshold:
            return "Medium"
        return "Low"

    def lane_context(self, origin: str, destination: str, k: int) -> dict | None:
        """Archetype of the lane implied by the selected origin/destination."""
        lane_id = f"{origin}-{destination}"
        idx = self.lane_index.get(lane_id)
        solution = self.art["clusters"]["solutions"].get(str(k))
        if idx is None or solution is None:
            return {"lane_id": lane_id, "known": False}
        cluster = solution["labels"][idx]
        summary = solution["summary"][cluster]
        row = self.lane_rows[idx]
        return {
            "lane_id": lane_id,
            "known": True,
            "cluster": cluster,
            "archetype": summary["archetype"],
            "lane_claim_rate": row["claim_rate"],
            "lane_avg_claim_value": row["avg_claim_value"],
            "lane_shipments": row["shipments"],
            "cluster_claim_rate": summary["claim_rate"],
            "cluster_expected_loss": summary["expected_loss_per_shipment"],
        }


STORE = ModelStore()


def ensure_plotly_asset() -> None:
    """
    Serve Plotly from the pip-installed package so the app works offline.

    A projector in a classroom is not a guaranteed internet connection, so the
    charting library is copied into static/vendor on first start instead of
    being pulled from a CDN at page load.
    """
    target = os.path.join(BASE_DIR, "static", "vendor", "plotly.min.js")
    if os.path.exists(target):
        return
    try:
        import plotly
        source = os.path.join(os.path.dirname(plotly.__file__),
                              "package_data", "plotly.min.js")
        if os.path.exists(source):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(source, target)
            print(f"[app] vendored plotly.min.js -> {target}")
    except Exception as exc:                  # pragma: no cover - defensive
        print(f"[app] could not vendor plotly.js ({exc}); "
              "the page will fall back to the CDN.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def presentation():
    return render_template("presentation.html",
                           generated_at=STORE.art["generated_at"])


# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
@app.route("/api/bootstrap")
def api_bootstrap():
    """Everything the presentation needs at load time, in one request."""
    art = STORE.art
    return jsonify({
        "generated_at": art["generated_at"],
        "dataset": art["dataset"],
        "features": art["features"],
        "leakage": art["leakage"],
        "eda": art["eda"],
        "classification": art["classification"],
        "best_classifier": art["best_classifier"],
        "coefficients": {"intercept": art["coefficients"]["intercept"],
                         "n_features": art["coefficients"]["n_features"],
                         "n_significant": art["coefficients"]["n_significant"],
                         "top": art["coefficients"]["top"]},
        "importances": art["importances"],
        "regression": art["regression"],
        "best_regressor": art["best_regressor"],
        "clusters": {"elbow": art["clusters"]["elbow"],
                     "default_k": art["clusters"]["default_k"],
                     "elbow_k": art["clusters"]["elbow_k"],
                     "best_k_silhouette": art["clusters"]["best_k_silhouette"],
                     "features": art["clusters"]["features"],
                     "feature_labels": art["clusters"]["feature_labels"],
                     "n_lanes": len(art["clusters"]["lanes"])},
        "controls": art["controls"],
        "cost_defaults": art["cost_defaults"],
        "baseline": art["baseline"],
        "carriers": art["carriers"],
    })


@app.route("/api/model-metrics")
def api_model_metrics():
    """Side-by-side comparison of the three classifiers and the two regressors."""
    art = STORE.art
    return jsonify({
        "classification": art["classification"],
        "best_classifier": art["best_classifier"],
        "regression": {k: {"key": k, "name": v["name"], "metrics": v["metrics"]}
                       for k, v in art["regression"].items()},
        "best_regressor": art["best_regressor"],
        "coefficients": art["coefficients"]["top"],
        "importances": art["importances"],
        "test_rows": art["dataset"]["test_rows"],
    })


@app.route("/api/threshold")
def api_threshold():
    """
    Recompute the confusion matrix and every derived rate at one threshold.

    Runs over the stored out-of-sample probability vector, so the ROC and PR
    curves stay fixed while the operating point moves along them.
    """
    model = request.args.get("model", STORE.best_classifier)
    if model not in STORE.eval_probs:
        return jsonify({"error": f"unknown model '{model}'"}), 400
    try:
        threshold = float(request.args.get("threshold", 0.5))
    except ValueError:
        return jsonify({"error": "threshold must be a number"}), 400
    threshold = min(max(threshold, 0.0), 1.0)

    result = confusion_at(STORE.y_test, STORE.eval_probs[model], threshold)
    meta = STORE.art["classification"][model]
    result.update({
        "model": model,
        "model_name": meta["name"],
        "roc_auc": meta["metrics"]["roc_auc"],
        "pr_auc": meta["metrics"]["pr_auc"],
        "base_rate": float(np.mean(STORE.y_test)),
    })
    return jsonify(result)


@app.route("/api/clusters")
def api_clusters():
    """One fitted K-Means solution: lane points, summary table, quality scores."""
    try:
        k = int(request.args.get("k", STORE.art["clusters"]["default_k"]))
    except ValueError:
        return jsonify({"error": "k must be an integer"}), 400

    solutions = STORE.art["clusters"]["solutions"]
    if str(k) not in solutions:
        return jsonify({"error": f"k must be one of {sorted(int(x) for x in solutions)}"}), 400

    solution = solutions[str(k)]
    lanes = STORE.art["clusters"]["lanes"]
    points = [{
        "lane_id": row["lane_id"],
        "claim_rate": row["claim_rate"],
        "avg_claim_value": row["avg_claim_value"],
        "shipments": row["shipments"],
        "expected_loss_per_shipment": row["expected_loss_per_shipment"],
        "avg_distance_km": row["avg_distance_km"],
        "avg_route_complexity": row["avg_route_complexity"],
        "cluster": solution["labels"][i],
    } for i, row in enumerate(lanes)]

    return jsonify({
        "k": k,
        "points": points,
        "summary": solution["summary"],
        "silhouette": solution["silhouette"],
        "inertia": solution["inertia"],
        "interpretation": solution["interpretation"],
        "elbow": STORE.art["clusters"]["elbow"],
        "elbow_k": STORE.art["clusters"]["elbow_k"],
        "best_k_silhouette": STORE.art["clusters"]["best_k_silhouette"],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Score one shipment with every model, and place it in its lane context."""
    payload = request.get_json(silent=True) or {}
    shipment = payload.get("shipment", {})
    threshold = float(payload.get("threshold", 0.5))
    k = int(payload.get("k", STORE.art["clusters"]["default_k"]))

    result = STORE.predict(shipment)
    primary = result["primary"]
    result["risk_band"] = STORE.risk_band(primary["probability"], threshold)
    result["risk_bands"] = {
        key: STORE.risk_band(p, threshold)
        for key, p in result["probabilities"].items()
    }
    result["threshold"] = threshold
    result["lane"] = STORE.lane_context(
        result["inputs"]["origin_city"], result["inputs"]["destination_city"], k)
    return jsonify(result)


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """
    Score a baseline shipment and a modified one, and report the difference.

    The difference is labelled a model-implied change on the page: it is what the
    fitted model predicts if that input were different, which is not the same as
    what would happen if the business actually changed it.
    """
    payload = request.get_json(silent=True) or {}
    current = payload.get("current", {})
    changes = payload.get("whatif", {})
    threshold = float(payload.get("threshold", 0.5))
    k = int(payload.get("k", STORE.art["clusters"]["default_k"]))

    scenario = dict(current)
    scenario.update(changes)

    before = STORE.predict(current)
    after = STORE.predict(scenario)

    def pack(res):
        p = res["primary"]
        return {
            "probabilities": res["probabilities"],
            "probability": p["probability"],
            "claim_value": p["claim_value"],
            "expected_loss": p["expected_loss"],
            "risk_band": STORE.risk_band(p["probability"], threshold),
        }

    b, a = pack(before), pack(after)
    return jsonify({
        "current": b,
        "whatif": a,
        "delta": {
            "probability_pp": (a["probability"] - b["probability"]) * 100,
            "probabilities_pp": {key: (a["probabilities"][key] - b["probabilities"][key]) * 100
                                 for key in a["probabilities"]},
            "claim_value": a["claim_value"] - b["claim_value"],
            "expected_loss": a["expected_loss"] - b["expected_loss"],
            "expected_loss_pct": ((a["expected_loss"] - b["expected_loss"]) / b["expected_loss"]
                                  if b["expected_loss"] > 0 else None),
        },
        "primary": before["primary"]["classifier"],
        "lane": STORE.lane_context(
            after["inputs"]["origin_city"], after["inputs"]["destination_city"], k),
        "label": "Model-implied change",
        "caveat": ("These are model-implied differences, not causal effects. The "
                   "model was fitted on observational shipment history."),
    })


@app.route("/api/eda")
def api_eda():
    """One categorical breakdown and one numeric relationship, with observations."""
    eda = STORE.art["eda"]
    category = request.args.get("category", eda["default_category"])
    numeric = request.args.get("numeric", eda["default_numeric"])
    if category not in eda["by_category"]:
        return jsonify({"error": f"unknown category '{category}'"}), 400
    if numeric not in eda["by_numeric"]:
        return jsonify({"error": f"unknown numeric variable '{numeric}'"}), 400
    return jsonify({
        "category": {"name": category, **eda["by_category"][category]},
        "numeric": {"name": numeric, **eda["by_numeric"][numeric]},
        "claim_rate": eda["claim_rate"],
    })


@app.route("/api/decision", methods=["POST"])
def api_decision():
    """
    Turn one scored shipment into ranked operational recommendations.

    The rules below are a decision framework - they compare this shipment's
    probability and claim size against the portfolio it came from and say what
    that combination usually warrants. They are not claims about causation.
    """
    payload = request.get_json(silent=True) or {}
    shipment = payload.get("shipment", {})
    threshold = float(payload.get("threshold", 0.5))
    k = int(payload.get("k", STORE.art["clusters"]["default_k"]))

    res = STORE.predict(shipment)
    primary = res["primary"]
    p = primary["probability"]
    value = primary["claim_value"]
    expected_loss = primary["expected_loss"]
    band = STORE.risk_band(p, threshold)
    lane = STORE.lane_context(res["inputs"]["origin_city"],
                              res["inputs"]["destination_city"], k)

    base_rate = float(np.mean(STORE.y_test))
    p_pct = float((STORE.eval_probs[STORE.best_classifier] < p).mean())
    v_pct = res["claim_value_percentile"]
    high_p = p >= threshold
    med_p = (not high_p) and p >= 0.6 * threshold
    high_v = v_pct >= 0.6
    declared = float(res["inputs"]["declared_value_rs"])

    recs: list[dict] = []
    if high_p and high_v:
        recs.append({
            "title": "Prioritise preventive intervention",
            "detail": (f"Claim probability {p*100:.1f}% is at or above the "
                       f"{threshold:.2f} action threshold, and the conditional claim "
                       f"value of ₹{value:,.0f} sits in the top "
                       f"{(1-v_pct)*100:.0f}% of the claim book. Expected loss is "
                       f"₹{expected_loss:,.0f} per shipment - the largest category "
                       f"of avoidable cost in this framework."),
            "tone": "critical",
        })
    elif high_p and not high_v:
        recs.append({
            "title": "Monitor and investigate systemic causes",
            "detail": (f"Claims are likely ({p*100:.1f}%) but individually small "
                       f"(₹{value:,.0f}). Per-shipment intervention rarely pays for "
                       f"itself here; the value is in finding the repeated cause, since "
                       f"expected loss is only ₹{expected_loss:,.0f} per shipment but "
                       f"recurs across the lane."),
            "tone": "warn",
        })
    elif (not high_p) and high_v:
        recs.append({
            "title": "Consider additional protection rather than inspection",
            "detail": (f"Probability is below the action threshold "
                       f"({p*100:.1f}% vs {threshold*100:.0f}%), but a claim would cost "
                       f"about ₹{value:,.0f}. Cover or upgraded packaging is usually "
                       f"cheaper than routine inspection at this frequency."),
            "tone": "info",
        })
    else:
        recs.append({
            "title": "Release under standard handling",
            "detail": (f"Probability {p*100:.1f}% and conditional claim value "
                       f"₹{value:,.0f} both sit below the intervention thresholds. "
                       f"Expected loss ₹{expected_loss:,.0f} is within normal "
                       f"operating cost."),
            "tone": "ok",
        })

    lever = STORE.best_lever(res["inputs"], expected_loss)
    if lever and lever["saving"] > 0:
        recs.append({
            "title": f"Biggest single lever: {lever['action']}",
            "detail": (f"Holding everything else fixed, the models put claim probability at "
                       f"{lever['new_probability']*100:.1f}% and expected loss at "
                       f"₹{lever['new_expected_loss']:,.0f} under that change — a model-implied "
                       f"saving of ₹{lever['saving']:,.0f} per shipment "
                       f"({lever['saving']/expected_loss*100:.0f}% of current exposure). "
                       f"Model-implied, not a causal guarantee."),
            "tone": "info",
        })
    elif lever:
        recs.append({
            "title": "No single input change lowers the modelled risk",
            "detail": (f"Each controllable lever tested (packaging quality, palletisation, "
                       f"utilisation, route complexity, loading quality) leaves expected loss at "
                       f"or above ₹{expected_loss:,.0f}. This shipment is already configured "
                       f"close to the model's best case for its route and contents."),
            "tone": "ok",
        })

    if med_p:
        recs.append({
            "title": "Borderline - worth a second look",
            "detail": (f"At {p*100:.1f}% this shipment is inside 60-100% of the "
                       f"{threshold:.2f} threshold. Lowering the threshold to "
                       f"{p:.2f} would bring it into scope; the simulator page shows "
                       f"what that costs in false positives."),
            "tone": "warn",
        })

    if lane and lane.get("known"):
        recs.append({
            "title": f"Lane sits in \"{lane['archetype']}\"",
            "detail": (f"{lane['lane_id']} has run {lane['lane_shipments']} shipments at a "
                       f"{lane['lane_claim_rate']*100:.1f}% historical claim rate "
                       f"(portfolio average {base_rate*100:.1f}%), average claim "
                       f"₹{lane['lane_avg_claim_value']:,.0f}. Its archetype carries an "
                       f"expected loss of ₹{lane['cluster_expected_loss']:,.0f} per "
                       f"shipment - lane-level action reaches every shipment on it, not "
                       f"just this one."),
            "tone": "info",
        })

    if declared > 0 and expected_loss / declared > 0.02:
        recs.append({
            "title": "Expected loss is material against declared value",
            "detail": (f"Expected loss ₹{expected_loss:,.0f} is "
                       f"{expected_loss/declared*100:.1f}% of the declared value "
                       f"₹{declared:,.0f}. Above roughly 2%, protection cost is "
                       f"usually easy to justify."),
            "tone": "warn",
        })

    takeaway = (
        f"This shipment carries a {p*100:.1f}% modelled claim probability "
        f"(higher than {p_pct*100:.0f}% of held-out shipments) and a conditional claim of "
        f"₹{value:,.0f}, giving an expected loss of ₹{expected_loss:,.0f}. "
        f"Risk band: {band} at a {threshold:.2f} threshold."
    )

    return jsonify({
        "probability": p,
        "claim_value": value,
        "expected_loss": expected_loss,
        "risk_band": band,
        "probability_percentile": p_pct,
        "claim_value_percentile": v_pct,
        "lane": lane,
        "recommendations": recs[:3],
        "best_lever": lever,
        "takeaway": takeaway,
        "classifier_name": primary["classifier_name"],
        "regressor_name": primary["regressor_name"],
    })


@app.route("/api/cost-benefit")
def api_cost_benefit():
    """
    Which threshold is cheapest, given what an intervention costs.

    Pure numpy over the stored evaluation arrays - no model is called, so this
    stays instant while a slider is being dragged. The two inputs are business
    assumptions, not data: `cost` is what acting on a flagged shipment costs, and
    `effectiveness` is the share of the damage that acting prevents.
    """
    model = request.args.get("model", STORE.best_classifier)
    if model not in STORE.eval_probs:
        return jsonify({"error": f"unknown model '{model}'"}), 400
    try:
        cost = float(request.args.get("cost", STORE.art["cost_defaults"]["cost_per_flagged"]))
        effectiveness = float(request.args.get(
            "effectiveness", STORE.art["cost_defaults"]["effectiveness"]))
    except ValueError:
        return jsonify({"error": "cost and effectiveness must be numbers"}), 400

    cost = max(cost, 0.0)
    effectiveness = min(max(effectiveness, 0.0), 1.0)

    probs = STORE.eval_probs[model]
    result = cb.compare(STORE.y_test, STORE.claim_amount_test, probs,
                        STORE.pred_value_test, cost, effectiveness)

    # Where the currently selected threshold sits on that same cost curve, so the
    # page can show what the present policy costs against the best available one.
    current = float(request.args.get("threshold", 0.5))
    current_cost = cb.total_cost(probs >= current, STORE.y_test,
                                 STORE.claim_amount_test, cost, effectiveness)

    result.update({
        "model": model,
        "model_name": STORE.art["classification"][model]["name"],
        "cost_per_flagged": cost,
        "effectiveness": effectiveness,
        "mean_claim": STORE.mean_claim,
        "closed_form_threshold": cb.closed_form_threshold(
            cost, effectiveness, STORE.mean_claim),
        "current_threshold": current,
        "current_cost": current_cost,
        "current_flagged_pct": float((probs >= current).mean()),
        "current_excess": current_cost - result["curve"]["optimal_cost"],
        "n": int(len(STORE.y_test)),
    })
    return jsonify(result)


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "models": sorted(STORE.classifiers) + sorted(f"reg_{k}" for k in STORE.regressors),
        "trained_at": STORE.art["generated_at"],
        "test_rows": int(len(STORE.y_test)),
    })


if __name__ == "__main__":
    ensure_plotly_asset()
    print("[app] Freight Damage Claims Analytics -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
