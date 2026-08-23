"""
Data loading, leakage control and the feature contract shared by every model.

The single most important idea in this module is the *leakage split*: the data
dictionary marks each variable with "Available Before Shipment". Anything that
only becomes known while the truck is moving - or after damage has already
happened - cannot be used to predict whether a claim will occur, because at the
moment the decision is made that information does not exist yet.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "freight_damage_claims.csv")
DICT_XLSX = os.path.join(BASE_DIR, "freight_damage_claims_dictionary.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TARGET_CLS = "damage_claim_raised"
TARGET_REG = "claim_amount_rs"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Columns dropped from the model matrix even though they are available before
# dispatch. Every reason below was verified against the data, not assumed.
DROPPED_WITH_REASON = {
    "shipment_id": "Row identifier - carries no signal.",
    "shipment_date": "Raw date; represented by shipment_month, day_of_week and season.",
    "shipment_quarter": "Redundant - derived from shipment_month.",
    "shipment_week": "Redundant - derived from shipment_date.",
    "financial_year": "Redundant - derived from shipment_date.",
    "lane_id": "210 levels; lane geography is carried by origin/destination city "
               "and historical_lane_claim_rate.",
    "origin_state": "Functionally determined by origin_city (verified 1:1).",
    "origin_zone": "Functionally determined by origin_city (verified 1:1).",
    "destination_state": "Functionally determined by destination_city (verified 1:1).",
    "destination_zone": "Functionally determined by destination_city (verified 1:1).",
    "volume_m3": "Exactly weight_kg / density_kg_per_m3 in 100% of rows.",
    "overloading_flag": "Exactly (vehicle_utilization_pct > 1) in 100% of rows.",
    "vehicle_capacity_kg": "Functionally determined by vehicle_type (verified 1:1).",
    "carrier_id": "Near-collinear with carrier_type plus the three numeric carrier "
                  "quality attributes, which are kept instead.",
}

# Business grouping used by the data page and the what-if panel.
FEATURE_GROUPS = {
    "Shipment": ["distance_km", "weight_kg", "declared_value_rs", "density_kg_per_m3",
                 "package_count", "fragility_class", "product_category",
                 "shipment_priority", "insurance_flag", "temperature_control_required"],
    "Carrier": ["carrier_type", "carrier_experience_years",
                "carrier_historical_damage_rate", "carrier_on_time_performance_pct",
                "driver_experience_years"],
    "Packaging": ["packaging_type", "packaging_quality_score", "packaging_age_days",
                  "palletized", "loading_quality_score"],
    "Vehicle": ["vehicle_type", "vehicle_age_years", "vehicle_utilization_pct"],
    "Route & Timing": ["origin_city", "destination_city", "transport_mode",
                       "route_complexity_score", "historical_lane_claim_rate",
                       "customer_handling_risk_score", "dispatch_delay_hours",
                       "season", "day_of_week", "shipment_month"],
}

# Readable labels for the presentation layer.
PRETTY = {
    "distance_km": "Distance (km)",
    "weight_kg": "Weight (kg)",
    "declared_value_rs": "Declared value (Rs)",
    "density_kg_per_m3": "Density (kg/m3)",
    "package_count": "Package count",
    "fragility_class": "Fragility class",
    "product_category": "Product category",
    "shipment_priority": "Shipment priority",
    "insurance_flag": "Insurance purchased",
    "temperature_control_required": "Cold chain required",
    "carrier_type": "Carrier type",
    "carrier_experience_years": "Carrier experience (yrs)",
    "carrier_historical_damage_rate": "Carrier historical damage rate",
    "carrier_on_time_performance_pct": "Carrier on-time performance",
    "driver_experience_years": "Driver experience (yrs)",
    "packaging_type": "Packaging type",
    "packaging_quality_score": "Packaging quality (1-5)",
    "packaging_age_days": "Packaging age (days)",
    "palletized": "Palletized",
    "loading_quality_score": "Loading quality (1-5)",
    "vehicle_type": "Vehicle type",
    "vehicle_age_years": "Vehicle age (yrs)",
    "vehicle_utilization_pct": "Vehicle utilisation",
    "origin_city": "Origin city",
    "destination_city": "Destination city",
    "transport_mode": "Transport mode",
    "route_complexity_score": "Route complexity (1-10)",
    "historical_lane_claim_rate": "Historical lane claim rate",
    "customer_handling_risk_score": "Receiver handling risk (1-10)",
    "dispatch_delay_hours": "Dispatch delay (hrs)",
    "season": "Season",
    "day_of_week": "Day of week",
    "shipment_month": "Month of dispatch",
    "damage_claim_raised": "Claim raised",
    "claim_amount_rs": "Claim amount (Rs)",
}


def pretty(name: str) -> str:
    """Human readable label for a raw or one-hot encoded column name."""
    if name in PRETTY:
        return PRETTY[name]
    if "=" in name:
        base, level = name.split("=", 1)
        return f"{PRETTY.get(base, base)}: {level}"
    return name.replace("_", " ").capitalize()


def load_raw() -> pd.DataFrame:
    """Load the shipment level dataset."""
    df = pd.read_csv(DATA_CSV)
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    return df


def dictionary_split() -> dict:
    """
    Read the data dictionary and split every column by *when it becomes known*.

    Returns four disjoint lists: identifiers, pre-shipment predictors,
    in-transit predictors (known only after dispatch) and post-event columns
    (known only after damage has occurred), plus the two targets.
    """
    d = pd.read_excel(DICT_XLSX)
    d.columns = [c.strip() for c in d.columns]
    role = d["Predictor / Target"].astype(str).str.strip()
    avail = d["Available Before Shipment"].astype(str).str.strip()
    name = d["Variable Name"].astype(str).str.strip()
    desc = d["Description"].astype(str).str.strip()

    is_pred = role.eq("Predictor")
    return {
        "identifiers": name[role.eq("Identifier")].tolist(),
        "pre_shipment": name[is_pred & avail.eq("Yes")].tolist(),
        "in_transit": name[is_pred & avail.str.startswith("No")].tolist(),
        "post_event": name[role.eq("Post-Event")].tolist(),
        "targets": name[role.eq("Target")].tolist(),
        "descriptions": dict(zip(name, desc)),
        "availability": dict(zip(name, avail)),
    }


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Model features, split into numeric and categorical, after leakage control."""
    split = dictionary_split()
    keep = [c for c in split["pre_shipment"] if c not in DROPPED_WITH_REASON]
    numeric = [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in keep if c not in numeric]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """
    Median-impute + standardise numerics, mode-impute + one-hot categoricals.

    Only three model features carry missing values (packaging_quality_score,
    loading_quality_score, driver_experience_years - 2% each), but imputation is
    part of the pipeline so a live what-if request can never break on a gap.
    """
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                 sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encoded_feature_names(preprocessor: ColumnTransformer,
                          numeric: list[str], categorical: list[str]) -> list[str]:
    """Readable names for the columns produced by the fitted preprocessor."""
    names = list(numeric)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    for col, cats in zip(categorical, ohe.categories_):
        dropped = 1 if ohe.drop_idx_ is not None else 0
        for level in cats[dropped:]:
            names.append(f"{col}={level}")
    return names


def baseline_row(df: pd.DataFrame, numeric: list[str],
                 categorical: list[str]) -> dict:
    """A typical shipment: median for numerics, modal level for categoricals."""
    row = {}
    for c in numeric:
        row[c] = float(np.round(df[c].median(), 4))
    for c in categorical:
        row[c] = str(df[c].mode(dropna=True).iloc[0])
    return row


def to_frame(payload: dict, baseline: dict, numeric: list[str],
             categorical: list[str]) -> pd.DataFrame:
    """
    Turn a partial JSON payload from the browser into one model-ready row.

    Anything the user did not send falls back to the baseline shipment, so the
    interactive pages only need to expose the handful of levers that matter.
    """
    row = dict(baseline)
    for key, value in (payload or {}).items():
        if key not in row:
            continue
        if key in numeric:
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue
        else:
            row[key] = str(value)
    frame = pd.DataFrame([row])[numeric + categorical]
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame


def carrier_profiles(df: pd.DataFrame) -> list[dict]:
    """
    Typical attribute values per carrier.

    carrier_id itself is excluded from the model (collinear with carrier_type
    plus the numeric carrier attributes), so the carrier dropdown on the what-if
    page works by *setting those attributes* to the carrier's typical values -
    the selection changes real model inputs rather than a cosmetic label.
    """
    out = []
    for cid, g in df.groupby("carrier_id"):
        out.append({
            "carrier_id": cid,
            "carrier_type": g["carrier_type"].mode().iloc[0],
            "carrier_experience_years": float(round(g["carrier_experience_years"].median(), 1)),
            "carrier_historical_damage_rate": float(round(g["carrier_historical_damage_rate"].median(), 4)),
            "carrier_on_time_performance_pct": float(round(g["carrier_on_time_performance_pct"].median(), 4)),
            "shipments": int(len(g)),
            "observed_claim_rate": float(round(g[TARGET_CLS].mean(), 4)),
        })
    return sorted(out, key=lambda r: r["carrier_id"])
