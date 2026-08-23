/* ==========================================================================
   Presentation controller.

   Holds one shared scenario across the whole deck: the shipment you build on
   the claim-probability page is the same shipment priced on the claim-value
   page, simulated on the what-if page and turned into a recommendation on the
   final page. Every number shown is fetched from a Flask endpoint that runs the
   trained models - nothing is computed or approximated in the browser.
   ========================================================================== */

'use strict';

const state = {
  slide: 0,
  boot: null,
  threshold: 0.5,
  k: 4,
  shipment: {},
  whatif: {},
  thrModel: 'logistic',
  regModel: 'random_forest',
  impModel: 'random_forest',
  clusters: null,
  thresholdResult: null,
};

const CLASSIFIER_ORDER = ['logistic', 'random_forest', 'bagging'];
const CLASSIFIER_SHORT = {
  logistic: 'Logistic Regression',
  random_forest: 'Random Forest',
  bagging: 'Bagging',
};

const slides = Array.from(document.querySelectorAll('.slide'));
const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------ helpers --- */
const fmtPct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const fmtRs = (v) => '₹' + Math.round(v).toLocaleString('en-IN');
const fmtNum = (v, d = 0) => Number(v).toLocaleString('en-IN', {
  minimumFractionDigits: d, maximumFractionDigits: d,
});

const MONEY = ['declared_value_rs'];
const RATE = ['carrier_historical_damage_rate', 'carrier_on_time_performance_pct',
              'vehicle_utilization_pct', 'historical_lane_claim_rate'];

function formatControl(name, value) {
  if (MONEY.includes(name)) return fmtRs(value);
  if (RATE.includes(name)) return fmtPct(value, name === 'vehicle_utilization_pct' ? 0 : 1);
  if (name === 'weight_kg' || name === 'distance_km') return fmtNum(value) +
    (name === 'weight_kg' ? ' kg' : ' km');
  return Number(value) % 1 === 0 ? String(Number(value)) : Number(value).toFixed(1);
}

/** Fetch JSON, surfacing an error rather than silently showing stale numbers. */
async function api(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`${url} -> ${res.status} ${detail.slice(0, 160)}`);
  }
  return res.json();
}

function postJSON(url, body) {
  return api(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/**
 * Guard against out-of-order responses.
 *
 * Dragging a slider fires several requests; they can come back in any order, and
 * an older reply landing last would repaint the screen with stale numbers. Each
 * refresh takes a ticket before it asks and drops its result if a newer request
 * has already answered.
 */
const seq = {};
function ticket(name) {
  seq[name] = (seq[name] || 0) + 1;
  return seq[name];
}
const isCurrent = (name, token) => seq[name] === token;

/** Collapse a burst of slider events into one request. */
function debounce(fn, wait = 90) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function control(name) {
  return state.boot.controls.find((c) => c.name === name);
}

/* --------------------------------------------------------- navigation --- */
function buildNav() {
  const dots = $('dots');
  dots.innerHTML = '';
  slides.forEach((slide, i) => {
    const b = document.createElement('button');
    b.className = 'dot';
    b.type = 'button';
    b.textContent = String(i + 1);
    b.title = slide.dataset.title || `Slide ${i + 1}`;
    b.setAttribute('aria-label', `Go to slide ${i + 1}: ${slide.dataset.title || ''}`);
    b.addEventListener('click', () => goTo(i));
    dots.appendChild(b);
  });
  $('prev-btn').addEventListener('click', () => goTo(state.slide - 1));
  $('next-btn').addEventListener('click', () => goTo(state.slide + 1));

  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)
        && ['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
    switch (e.key) {
      case 'ArrowRight': case 'PageDown': case ' ': goTo(state.slide + 1); e.preventDefault(); break;
      case 'ArrowLeft': case 'PageUp': goTo(state.slide - 1); e.preventDefault(); break;
      case 'Home': goTo(0); e.preventDefault(); break;
      case 'End': goTo(slides.length - 1); e.preventDefault(); break;
      default: break;
    }
  });

  window.addEventListener('resize', debounce(() => {
    Charts.resizeVisible(slides[state.slide]);
  }, 150));

  window.addEventListener('hashchange', () => {
    const n = parseInt(location.hash.replace('#', ''), 10);
    if (!Number.isNaN(n)) goTo(n - 1, false);
  });
}

function goTo(index, updateHash = true) {
  const i = Math.max(0, Math.min(slides.length - 1, index));
  state.slide = i;
  slides.forEach((s, n) => s.classList.toggle('active', n === i));
  Array.from($('dots').children).forEach((d, n) => d.classList.toggle('on', n === i));
  $('counter').innerHTML = `<b>${i + 1}</b> / ${slides.length} · ${slides[i].dataset.title}`;
  $('prev-btn').disabled = i === 0;
  $('next-btn').disabled = i === slides.length - 1;
  if (updateHash) history.replaceState(null, '', `#${i + 1}`);
  // Plotly cannot size a chart inside a hidden element, so re-fit on reveal.
  requestAnimationFrame(() => Charts.resizeVisible(slides[i]));
}

/* --------------------------------------------------- page 2 · the data -- */
function renderDataPage() {
  const b = state.boot;
  const d = b.dataset;
  const eda = b.eda;

  $('title-footprint').textContent =
    `${fmtNum(eda.n_shipments)} shipments · ${eda.date_range[0]} to ${eda.date_range[1]} · ` +
    `${eda.n_lanes} lanes · ${eda.n_carriers} carriers · ` +
    `${fmtNum(eda.n_claims)} claims worth ${fmtRs(eda.total_claim_value)}`;

  $('data-lede').innerHTML =
    `${fmtNum(d.rows)} shipments and ${d.columns} recorded columns. After leakage control ` +
    `and removing exact duplicates of other columns, <strong>${b.leakage.n_used} variables</strong> ` +
    `feed the models. Sixty-nine columns become six business questions.`;

  const groups = b.features.groups;
  const labels = b.features.labels;
  const host = $('var-cards');
  host.innerHTML = '';
  Object.entries(groups).forEach(([group, cols]) => {
    const card = document.createElement('div');
    card.className = 'varcard';
    const items = cols.map((c) => `<li>${labels[c] || c}</li>`).join('');
    card.innerHTML = `<h4>${group}</h4><ul>${items}</ul>`;
    host.appendChild(card);
  });
  const target = document.createElement('div');
  target.className = 'varcard target';
  target.innerHTML = `<h4>Target</h4><ul>
      <li><strong>Claim / no claim</strong> — ${fmtPct(eda.claim_rate)} of shipments</li>
      <li><strong>Claim value</strong> — median ${fmtRs(eda.claim_amount_stats.median)}</li>
      <li>Post-event severity and damage type are <em>excluded</em></li></ul>`;
  host.appendChild(target);

  const lk = b.leakage;
  $('pill-pre').textContent = `${lk.n_used} variables used`;
  $('pill-transit').textContent = `${lk.in_transit.length} in-transit`;
  $('pill-post').textContent = `${lk.post_event.length} post-event`;
  $('leakage-rule').textContent = lk.rule;
  const dropped = lk.pre_shipment.filter((v) => !v.used);
  $('leakage-examples').textContent =
    'Excluded as leakage: ' + lk.in_transit.slice(0, 4).map((v) => v.name).join(', ') +
    ', … + ' + lk.post_event.map((v) => v.name).join(', ') +
    `. Also dropped as redundant: ${dropped.length} pre-shipment columns ` +
    `(e.g. ${dropped.slice(0, 3).map((v) => v.name).join(', ')}).`;

  Charts.targetDonut(eda.target_counts, eda.claim_rate);
  Charts.monthlyLine(eda.monthly);
}

/* ---------------------------------------------------------- page 3 · eda */
function renderEdaControls() {
  const eda = state.boot.eda;
  const cat = $('eda-cat');
  const num = $('eda-num');
  cat.innerHTML = Object.entries(eda.by_category)
    .map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  num.innerHTML = Object.entries(eda.by_numeric)
    .map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  cat.value = eda.default_category;
  num.value = eda.default_numeric;
  cat.addEventListener('change', refreshEda);
  num.addEventListener('change', refreshEda);
  Charts.edaTarget(eda.target_counts);
  Charts.edaAmount(eda.claim_amount_hist, eda.claim_amount_stats);
}

async function refreshEda() {
  const token = ticket('eda');
  const data = await api(`/api/eda?category=${$('eda-cat').value}&numeric=${$('eda-num').value}`);
  if (!isCurrent('eda', token)) return;
  $('eda-cat-name').textContent = data.category.label.toLowerCase();
  $('eda-num-name').textContent = data.numeric.label.toLowerCase();
  Charts.edaCategory(data.category, data.claim_rate);
  Charts.edaNumeric(data.numeric, data.claim_rate);

  const stats = state.boot.eda.claim_amount_stats;
  const observations = data.category.observations.concat(data.numeric.observations).concat([
    `Claim amounts are strongly right-skewed: median ${fmtRs(stats.median)} but a mean of ` +
    `${fmtRs(stats.mean)} and a maximum of ${fmtRs(stats.max)}, so averages alone understate ` +
    `the tail risk.`,
  ]);
  $('eda-obs').innerHTML = observations.map((o) => `<li>${o}</li>`).join('');
}

/* ----------------------------------------------------- page 4 · lanes --- */
function renderClusterControls() {
  const slider = $('k-slider');
  const c = state.boot.clusters;
  state.k = c.default_k;
  slider.value = state.k;
  $('k-out').textContent = state.k;
  $('k-choice-note').textContent =
    `elbow at K=${c.elbow_k}, best silhouette at K=${c.best_k_silhouette}`;
  slider.addEventListener('input', () => {
    state.k = parseInt(slider.value, 10);
    $('k-out').textContent = state.k;
    refreshClusters();
  });
}

async function refreshClusters() {
  const token = ticket('clusters');
  const data = await api(`/api/clusters?k=${state.k}`);
  if (!isCurrent('clusters', token)) return;
  state.clusters = data;
  $('k-sil').textContent = data.silhouette.toFixed(3);
  $('k-inertia').textContent = fmtNum(data.inertia);
  $('k-lanes').textContent = fmtNum(data.points.length);
  $('cluster-interp').textContent = data.interpretation;

  $('cluster-table').innerHTML = data.summary.map((c) => `
    <tr>
      <td><span class="swatch" style="background:${Charts.CAT[c.cluster % Charts.CAT.length]}"></span>
        ${c.archetype}</td>
      <td class="num">${c.lane_count}</td>
      <td class="num">${fmtPct(c.claim_rate)}</td>
      <td class="num">${fmtRs(c.avg_claim_value)}</td>
      <td class="num">${fmtRs(c.expected_loss_per_shipment)}</td>
    </tr>`).join('');

  $('cluster-examples').innerHTML = data.summary.map((c) => `
    <div class="kv" style="align-items:flex-start">
      <span><span class="swatch" style="background:${Charts.CAT[c.cluster % Charts.CAT.length]}"></span>
        ${c.archetype}</span>
      <span style="font-weight:500;text-align:right;color:var(--ink-soft)">
        ${c.example_lanes.join(' · ')}</span>
    </div>`).join('');

  Charts.clusterScatter(data.points, data.summary);
  Charts.elbowChart(data.elbow, state.k);
  scheduleScenarioRefresh();
}

/* ----------------------------------------- page 5 & 10 · scenario UI ---- */
const PRIMARY_SLIDERS = ['distance_km', 'declared_value_rs', 'weight_kg', 'fragility_class',
                         'packaging_quality_score', 'vehicle_utilization_pct',
                         'route_complexity_score', 'carrier_historical_damage_rate'];
const PRIMARY_SELECTS = ['product_category', 'packaging_type', 'transport_mode',
                         'vehicle_type', 'shipment_priority', 'palletized',
                         'origin_city', 'destination_city'];
const WHATIF_SLIDERS = ['packaging_quality_score', 'vehicle_utilization_pct',
                        'route_complexity_score', 'fragility_class'];
const WHATIF_SELECTS = ['palletized', 'packaging_type'];

function sliderRow(meta, value, onInput) {
  const wrap = document.createElement('div');
  wrap.className = 'control';
  const id = `ctl-${meta.name}-${Math.random().toString(36).slice(2, 7)}`;
  // Declared value, weight and package count span orders of magnitude. On a
  // linear track the median sits in the first few percent of the travel and the
  // control is unusable, so those sliders move through log space instead.
  const log = meta.scale === 'log';
  const clamped = Math.min(Math.max(value, meta.min), meta.max);
  const toSlider = (v) => (log ? Math.log10(Math.max(v, meta.min)) : v);
  const fromSlider = (v) => (log ? Math.pow(10, v) : v);
  const lo = toSlider(meta.min);
  const hi = toSlider(meta.max);
  const step = log ? (hi - lo) / 200 : meta.step;

  wrap.innerHTML = `
    <label for="${id}">${meta.label}</label>
    <output id="${id}-out">${formatControl(meta.name, clamped)}</output>
    <div class="slider-row">
      <input type="range" id="${id}" min="${lo}" max="${hi}"
             step="${step}" value="${toSlider(clamped)}">
    </div>`;
  const input = wrap.querySelector('input');
  const out = wrap.querySelector('output');
  input.addEventListener('input', () => {
    const v = fromSlider(parseFloat(input.value));
    out.textContent = formatControl(meta.name, v);
    onInput(Math.round(v * 1000) / 1000);
  });
  return wrap;
}

function selectRow(meta, value, onChange) {
  const wrap = document.createElement('div');
  wrap.className = 'control';
  const id = `ctl-${meta.name}-${Math.random().toString(36).slice(2, 7)}`;
  const binary = meta.kind === 'numeric' && meta.is_binary;
  const options = binary
    ? [{ v: 0, t: 'No' }, { v: 1, t: 'Yes' }]
    : meta.levels.map((l) => ({ v: l, t: l }));
  wrap.innerHTML = `
    <label for="${id}">${meta.label}</label><span></span>
    <div class="slider-row">
      <select id="${id}">${options.map((o) =>
        `<option value="${o.v}" ${String(o.v) === String(value) ? 'selected' : ''}>${o.t}</option>`
      ).join('')}</select>
    </div>`;
  const sel = wrap.querySelector('select');
  sel.addEventListener('change', () => {
    onChange(binary ? parseFloat(sel.value) : sel.value);
  });
  return wrap;
}

function carrierRow(current, onChange) {
  const wrap = document.createElement('div');
  wrap.className = 'control';
  const id = `ctl-carrier-${Math.random().toString(36).slice(2, 7)}`;
  wrap.innerHTML = `
    <label for="${id}">Carrier</label><span></span>
    <div class="slider-row">
      <select id="${id}">${state.boot.carriers.map((c) =>
        `<option value="${c.carrier_id}" ${c.carrier_id === current ? 'selected' : ''}>
           ${c.carrier_id} · ${c.carrier_type} · ${fmtPct(c.observed_claim_rate)} observed
         </option>`).join('')}</select>
    </div>`;
  wrap.querySelector('select').addEventListener('change', (e) => {
    const carrier = state.boot.carriers.find((c) => c.carrier_id === e.target.value);
    onChange(carrier);
  });
  return wrap;
}

function buildShipmentPanel() {
  const host = $('shipment-controls');
  host.innerHTML = '';
  PRIMARY_SLIDERS.forEach((name) => {
    const meta = control(name);
    if (!meta) return;
    host.appendChild(sliderRow(meta, state.shipment[name], (v) => {
      state.shipment[name] = v;
      scheduleScenarioRefresh();
    }));
  });
  host.appendChild(carrierRow(state.shipment.__carrier_id, (carrier) => {
    state.shipment.__carrier_id = carrier.carrier_id;
    state.shipment.carrier_type = carrier.carrier_type;
    state.shipment.carrier_experience_years = carrier.carrier_experience_years;
    state.shipment.carrier_historical_damage_rate = carrier.carrier_historical_damage_rate;
    state.shipment.carrier_on_time_performance_pct = carrier.carrier_on_time_performance_pct;
    buildShipmentPanel();
    scheduleScenarioRefresh();
  }));
  PRIMARY_SELECTS.forEach((name) => {
    const meta = control(name);
    if (!meta) return;
    host.appendChild(selectRow(meta, state.shipment[name], (v) => {
      state.shipment[name] = v;
      scheduleScenarioRefresh();
    }));
  });
}

function buildWhatIfPanel() {
  const host = $('whatif-controls');
  host.innerHTML = '';
  const value = (name) => (name in state.whatif ? state.whatif[name] : state.shipment[name]);
  WHATIF_SLIDERS.forEach((name) => {
    const meta = control(name);
    if (!meta) return;
    host.appendChild(sliderRow(meta, value(name), (v) => {
      state.whatif[name] = v;
      scheduleWhatIf();
    }));
  });
  WHATIF_SELECTS.forEach((name) => {
    const meta = control(name);
    if (!meta) return;
    host.appendChild(selectRow(meta, value(name), (v) => {
      state.whatif[name] = v;
      scheduleWhatIf();
    }));
  });
  host.appendChild(carrierRow(state.whatif.__carrier_id || state.shipment.__carrier_id,
    (carrier) => {
      state.whatif.__carrier_id = carrier.carrier_id;
      state.whatif.carrier_type = carrier.carrier_type;
      state.whatif.carrier_experience_years = carrier.carrier_experience_years;
      state.whatif.carrier_historical_damage_rate = carrier.carrier_historical_damage_rate;
      state.whatif.carrier_on_time_performance_pct = carrier.carrier_on_time_performance_pct;
      buildWhatIfPanel();
      scheduleWhatIf();
    }));
}

function shipmentPayload(obj) {
  const out = {};
  Object.entries(obj).forEach(([k, v]) => { if (!k.startsWith('__')) out[k] = v; });
  return out;
}

function resetShipment() {
  state.shipment = Object.assign({}, state.boot.baseline);
  const match = state.boot.carriers.find(
    (c) => c.carrier_type === state.shipment.carrier_type);
  state.shipment.__carrier_id = match ? match.carrier_id : state.boot.carriers[0].carrier_id;
  state.whatif = {};
}

/* -------------------------------------------- page 5 · claim probability */
async function refreshPrediction() {
  const token = ticket('predict');
  const data = await postJSON('/api/predict', {
    shipment: shipmentPayload(state.shipment),
    threshold: state.threshold,
    k: state.k,
  });
  if (!isCurrent('predict', token)) return;

  $('p-lr').textContent = fmtPct(data.probabilities.logistic);
  $('p-rf').textContent = fmtPct(data.probabilities.random_forest);
  $('p-lr-sub').textContent =
    `${fmtPct(data.percentiles.logistic, 0)} of test shipments score lower`;
  $('p-rf-sub').textContent =
    `${fmtPct(data.percentiles.random_forest, 0)} of test shipments score lower`;

  const band = data.risk_band;
  const bandEl = $('risk-band');
  bandEl.textContent = band;
  bandEl.className = 'stat-value xl ' +
    (band === 'High' ? 'neg' : band === 'Medium' ? 'warn-c' : 'pos');
  $('risk-sub').textContent =
    `${data.primary.classifier_name} at a ${state.threshold.toFixed(2)} threshold`;

  $('model-probs').innerHTML = CLASSIFIER_ORDER.map((key) => `
    <div class="kv"><span>${CLASSIFIER_SHORT[key]}</span>
      <span>${fmtPct(data.probabilities[key])} <span class="pill ${
        data.risk_bands[key] === 'High' ? 'bad'
          : data.risk_bands[key] === 'Medium' ? 'warn' : 'ok'
      }">${data.risk_bands[key]}</span></span>
    </div>`).join('');

  const lane = data.lane;
  $('lane-context').innerHTML = lane && lane.known ? `
      <div class="kv"><span>Lane</span><span>${lane.lane_id}</span></div>
      <div class="kv"><span>Archetype</span><span>${lane.archetype}</span></div>
      <div class="kv"><span>Historical claim rate</span><span>${fmtPct(lane.lane_claim_rate)}</span></div>
      <div class="kv"><span>Average claim on lane</span><span>${fmtRs(lane.lane_avg_claim_value)}</span></div>
      <div class="kv"><span>Shipments observed</span><span>${fmtNum(lane.lane_shipments)}</span></div>`
    : `<p class="card-note">${lane ? lane.lane_id : ''} has too few shipments to be
       profiled as a lane (lanes need at least 30 movements to be clustered).</p>`;

  $('p5-value').textContent = fmtRs(data.primary.claim_value);
  $('p5-el').textContent = fmtRs(data.primary.expected_loss);
  $('p5-thr').textContent = state.threshold.toFixed(2);

  const hist = state.boot.classification[state.boot.best_classifier].prob_hist;
  Charts.probDistribution(hist, data.primary.probability, 'this shipment');

  renderClaimValuePanel(data);
}

/* ----------------------------------------------- page 8 · claim value --- */
function renderRegressionPage() {
  const seg = $('reg-model-seg');
  seg.innerHTML = Object.entries(state.boot.regression).map(([key, m]) =>
    `<button type="button" data-key="${key}"
      class="${key === state.regModel ? 'on' : ''}">${m.name.replace(' Regressor', '')}</button>`
  ).join('');
  seg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    state.regModel = b.dataset.key;
    renderRegressionPage();
  }));

  const m = state.boot.regression[state.regModel];
  $('reg-mae').textContent = fmtRs(m.metrics.mae);
  $('reg-rmse').textContent = fmtRs(m.metrics.rmse);
  $('reg-r2').textContent = m.metrics.r2.toFixed(3);
  $('reg-n').textContent = `${fmtNum(m.metrics.n)} test claims`;
  Charts.actualVsPredicted(m.scatter);
  Charts.residualPlot(m.scatter);

  const drivers = state.boot.regression.linear.drivers;
  const labels = state.boot.features.labels;
  $('reg-drivers').innerHTML = `
    <p class="card-note" style="margin:0 0 .4rem">Linear model, rupees added to the expected
      claim per one standard deviation (or versus the reference level).</p>` +
    drivers.slice(0, 8).map((d) => {
      const parts = d.feature.split('=');
      const name = labels[parts[0]] || parts[0];
      const lvl = parts[1] ? `: ${parts[1]}` : '';
      const sign = d.coefficient >= 0 ? 'neg' : 'pos';
      return `<div class="kv"><span>${name}${lvl}</span>
        <span class="${sign}">${d.coefficient >= 0 ? '+' : '−'}${fmtRs(Math.abs(d.coefficient))}</span></div>`;
    }).join('');
}

function renderClaimValuePanel(prediction) {
  const p = prediction.primary;
  $('cv-value').textContent = fmtRs(p.claim_value);
  $('cv-context').textContent =
    `${p.regressor_name} · larger than ${fmtPct(prediction.claim_value_percentile, 0)} ` +
    `of observed claims`;
  $('cv-p').textContent = fmtPct(p.probability);
  $('cv-ev').textContent = fmtRs(p.claim_value);
  $('cv-el').textContent = fmtRs(p.expected_loss);
}

/* ------------------------------------------------- page 6/7 · threshold - */
function renderThresholdControls() {
  const seg = $('thr-model-seg');
  seg.innerHTML = CLASSIFIER_ORDER.map((key) =>
    `<button type="button" data-key="${key}"
       class="${key === state.thrModel ? 'on' : ''}">${CLASSIFIER_SHORT[key]}</button>`
  ).join('');
  seg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    state.thrModel = b.dataset.key;
    seg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    refreshThreshold();
  }));

  $('thr-best').addEventListener('click', () => {
    setThreshold(state.boot.classification[state.thrModel].sweep.best_f1_threshold);
  });

  [['thr-slider', 'thr-out'], ['thr-slider-2', 'thr-out-2']].forEach(([sid, oid]) => {
    const slider = $(sid);
    slider.value = state.threshold;
    $(oid).textContent = state.threshold.toFixed(2);
    slider.addEventListener('input', () => setThreshold(parseFloat(slider.value)));
  });
}

/** Single place that moves the threshold, so both sliders stay in step. */
function setThreshold(value) {
  state.threshold = Math.round(value * 100) / 100;
  $('thr-slider').value = state.threshold;
  $('thr-slider-2').value = state.threshold;
  $('thr-out').textContent = state.threshold.toFixed(2);
  $('thr-out-2').textContent = state.threshold.toFixed(2);
  scheduleThreshold();
}

async function refreshThreshold() {
  const model = state.thrModel;
  const token = ticket('threshold');
  const data = await api(`/api/threshold?model=${model}&threshold=${state.threshold}`);
  if (!isCurrent('threshold', token)) return;
  state.thresholdResult = data;

  $('cm-tp').textContent = fmtNum(data.tp);
  $('cm-fp').textContent = fmtNum(data.fp);
  $('cm-tn').textContent = fmtNum(data.tn);
  $('cm-fn').textContent = fmtNum(data.fn);
  $('cm-n').textContent = `${fmtNum(data.n)} held-out shipments`;

  $('m-prec').textContent = fmtPct(data.precision);
  $('m-rec').textContent = fmtPct(data.recall);
  $('m-spec').textContent = fmtPct(data.specificity);
  $('m-f1').textContent = data.f1.toFixed(3);
  $('m-acc').textContent = fmtPct(data.accuracy);
  $('m-auc').textContent = data.roc_auc.toFixed(3);
  $('m-prauc').textContent = data.pr_auc.toFixed(3);
  $('thr-flagged').textContent =
    `${fmtNum(data.flagged)} of ${fmtNum(data.n)} flagged (${fmtPct(data.flagged_pct)})`;

  const curves = state.boot.classification[model].curves;
  Charts.rocChart('chart-roc', curves, data);
  Charts.prChart('chart-pr', curves, data, data.base_rate);

  renderTradeoff(data).catch(fail);
  scheduleScenarioRefresh();
}

/** Page 7: same threshold, framed as the business trade-off around it. */
async function renderTradeoff(current) {
  const model = state.thrModel;
  const token = ticket('tradeoff');
  const step = 0.12;
  const lower = Math.max(0.05, Math.round((state.threshold - step) * 100) / 100);
  const higher = Math.min(0.90, Math.round((state.threshold + step) * 100) / 100);

  const [lo, hi] = await Promise.all([
    api(`/api/threshold?model=${model}&threshold=${lower}`),
    api(`/api/threshold?model=${model}&threshold=${higher}`),
  ]);
  if (!isCurrent('tradeoff', token)) return;

  $('lower-thr').textContent = lower.toFixed(2);
  $('higher-thr').textContent = higher.toFixed(2);
  const fill = (prefix, d) => {
    $(`${prefix}-flag`).textContent = `${fmtNum(d.flagged)} (${fmtPct(d.flagged_pct)})`;
    $(`${prefix}-rec`).textContent = fmtPct(d.recall);
    $(`${prefix}-prec`).textContent = fmtPct(d.precision);
    $(`${prefix}-fp`).textContent = fmtNum(d.fp);
    $(`${prefix}-fn`).textContent = fmtNum(d.fn);
  };
  fill('low', lo);
  fill('high', hi);

  Charts.tradeoffChart(state.boot.classification[model].sweep, state.threshold);
  $('tradeoff-note').textContent =
    `Curves are fixed model properties. Only the red line — the chosen threshold — moves.`;

  const extraCaught = lo.tp - hi.tp;
  const extraFalse = lo.fp - hi.fp;
  const costRatio = extraFalse > 0 ? extraFalse / Math.max(extraCaught, 1) : null;
  $('tradeoff-interp').innerHTML =
    `Moving from <strong>${higher.toFixed(2)}</strong> down to <strong>${lower.toFixed(2)}</strong> ` +
    `catches <strong>${fmtNum(extraCaught)}</strong> more real claims but raises ` +
    `<strong>${fmtNum(extraFalse)}</strong> more false alarms` +
    (costRatio ? ` — about <strong>${costRatio.toFixed(1)} extra inspections per additional ` +
      `claim caught</strong>. ` : '. ') +
    `A lower threshold is justified whenever missing a genuinely risky shipment costs more ` +
    `than ${costRatio ? costRatio.toFixed(1) : 'that many'} unnecessary inspections. ` +
    `At the current setting of ${state.threshold.toFixed(2)} the model flags ` +
    `${fmtPct(current.flagged_pct)} of shipments and finds ${fmtPct(current.recall)} of all claims.`;
}

/* ------------------------------------------- page 9 · model comparison -- */
function renderComparison() {
  const models = state.boot.classification;
  const order = ['logistic', 'random_forest', 'bagging'];
  const best = state.boot.best_classifier;

  $('cmp-table').innerHTML = order.map((key, i) => {
    const m = models[key].metrics;
    const on = key === best;
    return `<tr style="${on ? 'background:var(--accent-soft)' : ''}">
      <td><span class="swatch" style="background:${Charts.CAT[i % Charts.CAT.length]}"></span>
        ${models[key].name}${on ? ' <span class="pill">best PR-AUC</span>' : ''}</td>
      <td class="num">${fmtPct(m.accuracy)}</td>
      <td class="num">${fmtPct(m.precision)}</td>
      <td class="num">${fmtPct(m.recall)}</td>
      <td class="num">${m.f1.toFixed(3)}</td>
      <td class="num">${m.roc_auc.toFixed(4)}</td>
      <td class="num">${m.pr_auc.toFixed(4)}</td></tr>`;
  }).join('');

  const aucs = order.map((k) => models[k].metrics.roc_auc);
  const spread = Math.max.apply(null, aucs) - Math.min.apply(null, aucs);
  const bestName = models[best].name;
  $('cmp-lede').innerHTML =
    `All three classifiers were trained on the same ${state.boot.leakage.n_used} pre-shipment ` +
    `variables and scored on the same ${fmtNum(state.boot.dataset.test_rows)} held-out shipments.`;
  $('cmp-verdict').innerHTML =
    `ROC-AUC spans just ${spread.toFixed(4)} across the three models, so the non-linear methods ` +
    `find little structure the linear model misses. <strong>${bestName}</strong> takes the best ` +
    `PR-AUC (${models[best].metrics.pr_auc.toFixed(4)}) — and being additive and inspectable, it ` +
    `is the model to prefer at equal accuracy. Random Forest and Bagging are kept because they ` +
    `confirm the ceiling is in the data, not in the choice of model.`;

  $('cmp-reg-table').innerHTML = Object.entries(state.boot.regression).map(([key, m]) => {
    const on = key === state.boot.best_regressor;
    return `<tr style="${on ? 'background:var(--accent-soft)' : ''}">
      <td>${m.name}${on ? ' <span class="pill">best R²</span>' : ''}</td>
      <td class="num">${fmtRs(m.metrics.mae)}</td>
      <td class="num">${fmtRs(m.metrics.rmse)}</td>
      <td class="num">${m.metrics.r2.toFixed(3)}</td></tr>`;
  }).join('');

  const labels = state.boot.features.labels;
  $('coef-table').innerHTML = state.boot.coefficients.top.slice(0, 14).map((c) => {
    const parts = c.feature.split('=');
    const name = (labels[parts[0]] || parts[0]) + (parts[1] ? `: ${parts[1]}` : '');
    const sig = c.p_value < 0.001 ? '<0.001' : c.p_value.toFixed(3);
    return `<tr>
      <td>${name}</td>
      <td class="num">${c.coefficient >= 0 ? '+' : ''}${c.coefficient.toFixed(3)}</td>
      <td class="num ${c.odds_ratio >= 1 ? 'neg' : 'pos'}">${c.odds_ratio.toFixed(3)}</td>
      <td class="num" style="${c.significant ? 'font-weight:650' : 'color:var(--ink-faint)'}">${sig}</td>
    </tr>`;
  }).join('');
  $('coef-note').textContent =
    `${state.boot.coefficients.n_significant} of ${state.boot.coefficients.n_features} ` +
    `encoded coefficients are significant at the 5% level. An odds ratio above 1 raises the ` +
    `odds of a claim; numeric variables are standardised, so the effect is per one standard ` +
    `deviation. p-values are Wald tests from the observed information matrix.`;

  Charts.comparisonCurves(models, order);
}

/* ------------------------------------------------ page 10 · what-if ----- */
function renderDriverControls() {
  const seg = $('imp-seg');
  const options = [['random_forest', 'Random Forest'], ['bagging', 'Bagging'],
                   ['logistic', 'Logistic (|coef|)']];
  seg.innerHTML = options.map(([key, name]) =>
    `<button type="button" data-key="${key}"
       class="${key === state.impModel ? 'on' : ''}">${name}</button>`).join('');
  seg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    state.impModel = b.dataset.key;
    renderDriverControls();
  }));
  renderDrivers();
}

function renderDrivers() {
  const labels = state.boot.features.labels;
  let rows;
  let note;
  if (state.impModel === 'logistic') {
    const grouped = {};
    state.boot.coefficients.top.forEach((c) => {
      const key = c.feature.split('=')[0];
      grouped[key] = (grouped[key] || 0) + Math.abs(c.coefficient);
    });
    const total = Object.values(grouped).reduce((a, b) => a + b, 0) || 1;
    rows = Object.entries(grouped)
      .map(([variable, v]) => ({ variable, importance: v / total }))
      .sort((a, b) => b.importance - a.importance).slice(0, 10);
    note = 'Logistic Regression: absolute standardised coefficients, summed per variable. ' +
           'Direction is on the model-comparison page.';
  } else {
    rows = state.boot.importances[state.impModel].slice(0, 10);
    note = `${state.impModel === 'bagging' ? 'Bagging' : 'Random Forest'}: mean impurity ` +
           'decrease, summed back from one-hot columns to the original variable. Importance ' +
           'measures how much a variable is used, not the direction of its effect.';
  }
  const max = rows[0] ? rows[0].importance : 1;
  $('driver-bars').innerHTML = rows.map((r) => `
    <div class="bar-row">
      <span>${labels[r.variable] || r.variable}</span>
      <span class="bar-track"><span class="bar-fill"
        style="width:${(r.importance / max * 100).toFixed(1)}%"></span></span>
      <span class="num">${(r.importance * 100).toFixed(1)}%</span>
    </div>`).join('');
  $('driver-note').textContent = note;
}

async function refreshWhatIf() {
  const token = ticket('whatif');
  const data = await postJSON('/api/whatif', {
    current: shipmentPayload(state.shipment),
    whatif: shipmentPayload(state.whatif),
    threshold: state.threshold,
    k: state.k,
  });
  if (!isCurrent('whatif', token)) return;

  $('wi-current').textContent = fmtPct(data.current.probability);
  $('wi-new').textContent = fmtPct(data.whatif.probability);
  const delta = data.delta.probability_pp;
  const deltaEl = $('wi-delta');
  deltaEl.textContent = `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)} pp`;
  deltaEl.className = 'stat-value ' + (Math.abs(delta) < 0.05 ? '' : delta > 0 ? 'neg' : 'pos');

  $('wi-lr').textContent = fmtPct(data.whatif.probabilities.logistic);
  $('wi-rf').textContent = fmtPct(data.whatif.probabilities.random_forest);
  $('wi-value').textContent = fmtRs(data.whatif.claim_value);
  $('wi-el').textContent = fmtRs(data.whatif.expected_loss);
  $('wi-caveat').textContent = ' ' + data.caveat;
}

/* ----------------------------------------------- page 11 · decision ----- */
async function refreshDecision() {
  const token = ticket('decision');
  const data = await postJSON('/api/decision', {
    shipment: shipmentPayload(state.shipment),
    threshold: state.threshold,
    k: state.k,
  });
  if (!isCurrent('decision', token)) return;

  $('dec-p').textContent = fmtPct(data.probability);
  $('dec-p-pct').textContent =
    `higher than ${fmtPct(data.probability_percentile, 0)} of test shipments`;
  $('dec-v').textContent = fmtRs(data.claim_value);
  $('dec-v-pct').textContent =
    `larger than ${fmtPct(data.claim_value_percentile, 0)} of observed claims`;
  $('dec-loss').textContent = fmtRs(data.expected_loss);
  const bandEl = $('dec-band');
  bandEl.textContent = data.risk_band;
  bandEl.className = 'stat-value sm ' +
    (data.risk_band === 'High' ? 'neg' : data.risk_band === 'Medium' ? 'warn-c' : 'pos');
  $('dec-band-sub').textContent = `at a ${state.threshold.toFixed(2)} threshold`;

  $('dec-archetype').textContent = data.lane && data.lane.known
    ? data.lane.archetype : 'Lane not profiled';
  $('dec-prob').textContent = `Claim probability ${fmtPct(data.probability)}`;
  $('dec-sev').textContent = `Claim severity ${fmtRs(data.claim_value)}`;
  $('dec-el').textContent = `Expected loss ${fmtRs(data.expected_loss)}`;

  $('dec-recs').innerHTML = data.recommendations.map((r) => `
    <div class="rec ${r.tone}">
      <p class="rec-title">${r.title}</p>
      <p class="rec-detail">${r.detail}</p>
    </div>`).join('');

  $('dec-takeaway').textContent = data.takeaway;
  $('dec-footer').textContent =
    `Probability from ${data.classifier_name}; claim value from ${data.regressor_name}; ` +
    `lane archetype from the K=${state.k} K-Means solution. Recommendations are a decision ` +
    `framework applied to this scenario, not causal claims about the dataset.`;
}

/* ---------------------------------------------------------- scheduling -- */
const scheduleThreshold = debounce(() => refreshThreshold().catch(fail), 70);
const scheduleWhatIf = debounce(() => refreshWhatIf().catch(fail), 90);
const scheduleScenarioRefresh = debounce(() => {
  Promise.all([refreshPrediction(), refreshWhatIf(), refreshDecision()]).catch(fail);
}, 110);

function fail(err) {
  console.error(err);
  const el = $('loading');
  el.classList.remove('hidden');
  el.innerHTML = `<span class="err">Could not reach the model server: ${err.message}<br>
    Run <span class="mono">python train_models.py</span> then
    <span class="mono">python app.py</span> and reload.</span>`;
}

/* ---------------------------------------------------------------- boot -- */
async function boot() {
  state.boot = await api('/api/bootstrap');
  state.threshold = 0.5;
  state.thrModel = state.boot.best_classifier;
  state.regModel = state.boot.best_regressor;
  resetShipment();

  buildNav();
  renderDataPage();
  renderEdaControls();
  renderClusterControls();
  renderThresholdControls();
  renderComparison();
  renderRegressionPage();
  renderDriverControls();
  buildShipmentPanel();
  buildWhatIfPanel();

  $('reset-shipment').addEventListener('click', () => {
    resetShipment();
    buildShipmentPanel();
    buildWhatIfPanel();
    scheduleScenarioRefresh();
  });
  $('reset-whatif').addEventListener('click', () => {
    state.whatif = {};
    buildWhatIfPanel();
    scheduleWhatIf();
  });

  await Promise.all([refreshEda(), refreshClusters(), refreshThreshold(), refreshWhatIf()]);

  $('loading').classList.add('hidden');
  const start = parseInt(location.hash.replace('#', ''), 10);
  goTo(Number.isNaN(start) ? 0 : start - 1, false);
}

boot().catch(fail);
