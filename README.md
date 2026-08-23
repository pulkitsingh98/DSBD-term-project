# Freight Damage Claims Analytics

**Predicting Risk. Estimating Loss. Improving Decisions.**

An interactive analytics presentation, built as a Flask application, over 25,000
Indian freight shipments (Jan 2023 – Dec 2025). Eleven landscape slides move from
the business problem to an operating decision rule, and every number on every
slide is computed by a trained model — nothing is hardcoded, mocked, or
illustrative.

## Running it

```bash
pip install -r requirements.txt
python train_models.py      # ~15 s, writes models/
python app.py               # http://127.0.0.1:5000
```

Training is a separate step on purpose: the app only loads saved artefacts, so
starting the server is instant and the numbers never shift between runs.
`models/` is gitignored (≈54 MB), so run `train_models.py` once after cloning.

Navigate with **← / →**, **Home**, **End**, the page numbers, or the Previous /
Next buttons. The deck is designed for a 16:9 projector and never scrolls.
Plotly is served from `static/vendor/`, copied out of the installed `plotly`
package on first start, so the presentation works with no internet connection.

## The slides

| # | Slide | What is interactive |
|---|-------|---------------------|
| 1 | Problem statement | — |
| 2 | Data → business variables | — |
| 3 | Exploratory analysis | Variable dropdowns redraw two charts and regenerate the observations |
| 4 | Lane risk archetypes | K slider (2–6) re-clusters, retitles and rescores |
| 5 | Claim probability | 17 shipment controls re-score three classifiers live |
| 6 | Threshold simulator | Threshold slider moves the operating point on fixed ROC / PR curves |
| 7 | Why the threshold matters | Same threshold, framed as an inspection-cost trade-off |
| 8 | Claim value | Model toggle; expected loss for the current shipment |
| 9 | Model comparison | Logistic vs Random Forest vs Bagging on identical features |
| 10 | Risk drivers + what-if | Importance source toggle; what-if levers vs the current shipment |
| 11 | Business decision | Three recommendations regenerated from the live scenario |

## The models

All three classifiers are trained on the **same** feature matrix, so the
comparison on slide 9 is fair.

| Model | Purpose | Test result |
|-------|---------|-------------|
| Logistic Regression | Claim probability, interpretable | ROC-AUC 0.706 · PR-AUC 0.306 |
| Random Forest | Claim probability, non-linear | ROC-AUC 0.699 · PR-AUC 0.298 |
| Bagging (decision trees) | Claim probability, variance reduction | ROC-AUC 0.701 · PR-AUC 0.301 |
| Linear Regression | Claim value, given a claim | MAE ₹13,378 · RMSE ₹29,015 · R² 0.172 |
| Random Forest Regressor | Claim value benchmark | MAE ₹12,265 · RMSE ₹27,960 · R² 0.231 |
| K-Means | Lane risk archetypes | 210 lanes, K chosen by elbow (K=4) and silhouette |

Held out 25% of rows (6,250 shipments) with stratification, `random_state=42`.

Three findings worth stating plainly rather than dressing up:

- **The three classifiers are within 0.007 ROC-AUC of each other.** The
  non-linear methods find almost nothing the additive model misses, which says
  the ceiling is in the data, not in the choice of algorithm. Logistic
  Regression wins on PR-AUC and is the one to prefer at equal accuracy.
- **At the conventional 0.50 threshold the model flags 0.8% of shipments and
  catches 2.9% of claims.** That is what a 14.9% base rate does to a default
  cut-off, and it is the entire point of slides 6 and 7. The F1-optimal
  threshold is 0.15.
- **Claim value is only moderately predictable (R² 0.231).** How much a claim
  costs depends heavily on how badly the goods were damaged, and severity is a
  post-event variable the leakage rule forbids. A moderate R² is the expected
  result here, not a modelling failure.

## Leakage control

The data dictionary marks every column with *Available Before Shipment*. That
marking drives the feature set, read at training time rather than hardcoded:

- **46 pre-shipment predictors** — known when the shipment decision is made.
- **17 in-transit predictors** — temperature, rainfall, road condition, delivery
  delay, warehouse handling, the handling chain. Excluded: they do not exist yet
  at the moment of prediction.
- **3 post-event columns** — `damage_severity_score`, `damage_type`,
  `claim_processing_days`. Excluded: they exist only because damage already
  happened. All three are missing for exactly the 85.06% of rows with no claim.

A further 13 of those pre-shipment columns are dropped as redundant, each
verified against the data rather than assumed:

- `weight_kg = volume_m3 × density_kg_per_m3` in **100%** of rows → drop `volume_m3`
- `overloading_flag = (vehicle_utilization_pct > 1)` in **100%** of rows → drop the flag
- `origin_city` determines state and zone 1:1 → drop both, twice over
- `vehicle_type` determines `vehicle_capacity_kg` 1:1 → drop the capacity
- `carrier_id` is near-collinear with `carrier_type` plus the three numeric
  carrier-quality attributes → keep the attributes, drop the id

That leaves **33 model features** (23 numeric, 10 categorical) expanding to 87
encoded columns. Missing values are confined to three of them
(`packaging_quality_score`, `loading_quality_score`, `driver_experience_years`,
2% each) and are median-imputed inside the pipeline.

`carrier_id` still drives the carrier dropdown on slides 5 and 10: selecting a
carrier sets its typical damage rate, on-time performance and experience, so the
choice changes real model inputs rather than a cosmetic label.

## Layout

```
app.py                  Flask server, JSON APIs, model store
train_models.py         one offline run: train, evaluate, pre-compute, save
src/
    preprocessing.py    leakage split, feature contract, encoding pipeline
    classification.py   logistic / random forest / bagging, curves, Wald p-values
    regression.py       claim value models and their diagnostics
    clustering.py       lane aggregation, K-Means, archetype naming
templates/
    base.html           page shell
    presentation.html   all eleven slides
static/
    css/style.css       one stylesheet, viewport-relative sizing
    js/charts.js        Plotly theme and every figure
    js/presentation.js  navigation, shared scenario state, API calls
models/                 saved pipelines + artifacts.json (generated)
```

### API

| Endpoint | Returns |
|----------|---------|
| `GET /api/bootstrap` | Everything the deck needs at load |
| `GET /api/model-metrics` | Classifier and regressor comparison |
| `GET /api/threshold?model=&threshold=` | Confusion matrix and rates, recomputed from the held-out probability vector |
| `GET /api/clusters?k=` | One fitted K-Means solution: lane points, summary, silhouette, inertia |
| `GET /api/eda?category=&numeric=` | One breakdown, one relationship, with generated observations |
| `POST /api/predict` | Three classifiers, two regressors, expected loss, lane context |
| `POST /api/whatif` | Baseline vs modified scenario and the model-implied delta |
| `POST /api/decision` | Three ranked recommendations and a takeaway |
| `GET /api/health` | Loaded models and training timestamp |

## Two design notes

**Archetype names are derived, not written.** Each K-Means cluster is named from
its own standardised profile — claim frequency, claim size, operating
difficulty, volume — so the labels describe whatever the algorithm actually
found at that K. Changing K changes the names.

**"Model-implied" is not "causal".** The what-if page and the lever
recommendation on the final slide report what the fitted models predict if an
input were different. The models were fitted on observational shipment history,
so that is an association, and both pages say so on screen.
