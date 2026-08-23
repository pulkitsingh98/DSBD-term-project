"""
Lane risk archetypes.

The unit of analysis here is the *lane*, not the shipment: a lane is an
origin-destination pair that the network runs repeatedly, and it is the level at
which an operations team can actually change something (re-route, re-package,
re-tender to a different carrier). Shipment rows are therefore aggregated into
one row per lane before K-Means is run.

Archetype names are derived from the standardised profile of each fitted
cluster, so they describe whatever the algorithm actually found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import RANDOM_STATE, TARGET_CLS, TARGET_REG

# Lane profile used for clustering. Mixes demand (volume), outcome (claim rate,
# claim size) and operating difficulty (distance, handling, complexity).
LANE_FEATURES = [
    "shipments",
    "claim_rate",
    "avg_claim_value",
    "avg_distance_km",
    "avg_declared_value_rs",
    "avg_fragility_class",
    "avg_packaging_quality",
    "avg_route_complexity",
    "avg_handling_touchpoints",
]

# Money variables are strongly right-skewed across lanes (avg claim value has a
# skew above 5), so a handful of lanes would otherwise sit so far from everyone
# else that K-Means spends its clusters isolating them. Log scaling these two
# before standardisation keeps the distance metric meaningful.
LOG_SCALED = ["avg_claim_value", "avg_declared_value_rs"]

LANE_LABELS = {
    "shipments": "Shipment volume",
    "claim_rate": "Claim rate",
    "avg_claim_value": "Avg claim value",
    "avg_distance_km": "Avg distance (km)",
    "avg_declared_value_rs": "Avg declared value",
    "avg_fragility_class": "Avg fragility",
    "avg_packaging_quality": "Avg packaging quality",
    "avg_route_complexity": "Avg route complexity",
    "avg_handling_touchpoints": "Avg handling touchpoints",
}


def build_lane_table(df: pd.DataFrame, min_shipments: int = 30) -> pd.DataFrame:
    """
    One row per lane.

    Lanes with very few shipments are dropped: a claim rate estimated from a
    handful of movements is noise, and it would dominate the clustering.
    """
    claims = df[df[TARGET_CLS] == 1]
    claim_value = claims.groupby("lane_id")[TARGET_REG].mean()

    lanes = df.groupby("lane_id").agg(
        origin_city=("origin_city", "first"),
        destination_city=("destination_city", "first"),
        shipments=("shipment_id", "size"),
        claims=(TARGET_CLS, "sum"),
        claim_rate=(TARGET_CLS, "mean"),
        avg_distance_km=("distance_km", "mean"),
        avg_declared_value_rs=("declared_value_rs", "mean"),
        avg_fragility_class=("fragility_class", "mean"),
        avg_packaging_quality=("packaging_quality_score", "mean"),
        avg_route_complexity=("route_complexity_score", "mean"),
        avg_handling_touchpoints=("handling_touchpoints", "mean"),
    )
    lanes["avg_claim_value"] = claim_value.reindex(lanes.index).fillna(0.0)
    lanes["expected_loss_per_shipment"] = lanes["claim_rate"] * lanes["avg_claim_value"]
    lanes = lanes[lanes["shipments"] >= min_shipments].copy()
    return lanes.reset_index()


def _archetype_name(profile: dict) -> str:
    """Name a cluster from its standardised profile (z-scores across clusters)."""
    freq = profile["claim_rate"]
    value = profile["avg_claim_value"]
    ops = (profile["avg_route_complexity"] + profile["avg_handling_touchpoints"]
           + profile["avg_distance_km"]) / 3.0
    volume = profile["shipments"]
    hi, lo = 0.45, -0.35

    if freq > hi and value > hi:
        return "High-Risk, High-Exposure Lanes"
    if freq > hi:
        return "High-Claim-Frequency Lanes"
    if value > hi:
        return "High-Value Exposure Lanes"
    if ops > hi:
        return "Operationally Difficult Lanes"
    if freq < lo and value < lo:
        return "Stable Low-Risk Lanes"
    if freq < lo:
        return "Low-Frequency, Watchlist Lanes"
    if volume > hi:
        return "High-Volume Core Lanes"
    return "Moderate Mixed-Risk Lanes"


def _dedupe(names: list[str], profiles: list[dict]) -> list[str]:
    """Keep archetype names unique by adding the strongest secondary trait."""
    out = list(names)
    seen: dict[str, list[int]] = {}
    for i, n in enumerate(out):
        seen.setdefault(n, []).append(i)
    for name, idxs in seen.items():
        if len(idxs) == 1:
            continue
        for i in idxs:
            trait = max(profiles[i], key=lambda k: abs(profiles[i][k]))
            direction = "high" if profiles[i][trait] >= 0 else "low"
            out[i] = f"{name} ({direction} {LANE_LABELS[trait].lower()})"
    return out


def cluster_lanes(lanes: pd.DataFrame, k_values=range(2, 7)) -> dict:
    """
    Fit K-Means for every K in the allowed range and describe each solution.

    Every K is fitted once at training time and cached, so moving the K slider in
    the browser is an instant lookup of a real fitted model rather than a
    re-fit or an approximation.
    """
    X = lanes[LANE_FEATURES].astype(float).copy()
    for col in LOG_SCALED:
        X[col] = np.log1p(X[col])
    Z = StandardScaler().fit_transform(X.to_numpy(dtype=float))

    solutions, elbow = {}, []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE).fit(Z)
        labels = km.labels_
        sil = float(silhouette_score(Z, labels)) if k > 1 else float("nan")
        elbow.append({"k": int(k), "inertia": float(km.inertia_), "silhouette": sil})

        # Cluster profile in standardised units, used only for naming.
        profiles = []
        for c in range(k):
            mask = labels == c
            profiles.append({f: float(Z[mask, i].mean())
                             for i, f in enumerate(LANE_FEATURES)})
        names = _dedupe([_archetype_name(p) for p in profiles], profiles)

        # Present clusters ordered by expected loss, so cluster 0 is always the
        # calmest group on screen regardless of the arbitrary K-Means labelling.
        raw_order = sorted(
            range(k),
            key=lambda c: float(lanes.loc[labels == c, "expected_loss_per_shipment"].mean()),
        )
        remap = {old: new for new, old in enumerate(raw_order)}
        ordered_labels = np.array([remap[l] for l in labels])

        summary = []
        for c in range(k):
            mask = ordered_labels == c
            sub = lanes[mask]
            summary.append({
                "cluster": int(c),
                "archetype": names[raw_order[c]],
                "lane_count": int(mask.sum()),
                "shipments": int(sub["shipments"].sum()),
                "claim_rate": float(sub["claim_rate"].mean()),
                "avg_claim_value": float(sub["avg_claim_value"].mean()),
                "expected_loss_per_shipment": float(sub["expected_loss_per_shipment"].mean()),
                "avg_distance_km": float(sub["avg_distance_km"].mean()),
                "avg_declared_value_rs": float(sub["avg_declared_value_rs"].mean()),
                "avg_fragility_class": float(sub["avg_fragility_class"].mean()),
                "avg_packaging_quality": float(sub["avg_packaging_quality"].mean()),
                "avg_route_complexity": float(sub["avg_route_complexity"].mean()),
                "avg_handling_touchpoints": float(sub["avg_handling_touchpoints"].mean()),
                "profile_z": profiles[raw_order[c]],
                "example_lanes": sub.nlargest(3, "shipments")["lane_id"].tolist(),
            })

        solutions[str(k)] = {
            "k": int(k),
            "labels": ordered_labels.tolist(),
            "silhouette": sil,
            "inertia": float(km.inertia_),
            "summary": summary,
            "interpretation": _interpret(summary, sil),
        }

    return {
        "features": LANE_FEATURES,
        "feature_labels": LANE_LABELS,
        "log_scaled": LOG_SCALED,
        "lanes": lanes.to_dict(orient="records"),
        "elbow": elbow,
        "best_k_silhouette": max(elbow, key=lambda e: e["silhouette"])["k"],
        "elbow_k": _elbow_k(elbow),
        "default_k": _elbow_k(elbow),
        "solutions": solutions,
    }


def _elbow_k(elbow: list[dict]) -> int:
    """
    Pick K at the knee of the inertia curve.

    Measures each point's perpendicular distance from the straight line joining
    the first and last point of the curve and takes the furthest one - the
    standard geometric reading of the elbow, done numerically instead of by eye.
    """
    ks = np.array([e["k"] for e in elbow], dtype=float)
    inertia = np.array([e["inertia"] for e in elbow], dtype=float)
    if len(ks) < 3:
        return int(ks[0])
    p0 = np.array([ks[0], inertia[0]])
    p1 = np.array([ks[-1], inertia[-1]])
    line = p1 - p0
    line = line / np.linalg.norm(line)
    dists = []
    for k, i in zip(ks, inertia):
        v = np.array([k, i]) - p0
        dists.append(np.linalg.norm(v - np.dot(v, line) * line))
    return int(ks[int(np.argmax(dists))])


def _interpret(summary: list[dict], silhouette: float) -> str:
    """A sentence describing this particular solution, built from its numbers."""
    calm = summary[0]
    worst = summary[-1]
    spread = (worst["expected_loss_per_shipment"] / calm["expected_loss_per_shipment"]
              if calm["expected_loss_per_shipment"] > 0 else float("inf"))
    spread_txt = (f"{spread:.1f}x" if np.isfinite(spread) else "many times")
    return (
        f"With K={len(summary)} the lane network splits into groups whose expected loss "
        f"per shipment differs by about {spread_txt}. The calmest group "
        f"(\"{calm['archetype']}\", {calm['lane_count']} lanes) runs a "
        f"{calm['claim_rate']*100:.1f}% claim rate at an average claim of "
        f"₹{calm['avg_claim_value']:,.0f}, while the most exposed group "
        f"(\"{worst['archetype']}\", {worst['lane_count']} lanes) runs "
        f"{worst['claim_rate']*100:.1f}% at ₹{worst['avg_claim_value']:,.0f}. "
        f"Silhouette score {silhouette:.3f}."
    )
