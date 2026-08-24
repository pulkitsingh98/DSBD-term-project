/* ==========================================================================
   Chart layer.

   One Plotly theme for the whole deck: light or dark ground, hairline axes, a
   single accent, and a categorical palette used only where categories genuinely
   need separating (clusters, model comparison). Every figure answers one
   question, so titles live in the surrounding card and never inside the plot.

   Colours are never written here. They are read from the CSS custom properties
   on :root at draw time, which is what lets the same code serve both themes.
   ========================================================================== */

const FONT_FAMILY =
  'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
const CONFIG = { displayModeBar: false, responsive: true, doubleClick: 'reset' };

/**
 * Read the live theme out of CSS instead of duplicating it here.
 *
 * Every colour is already a custom property on :root, and the dark theme just
 * redefines those same names. Reading them at draw time means a chart is
 * theme-correct by construction - there is no second palette to keep in sync.
 */
let PALETTE = null;
function palette() {
  if (PALETTE) return PALETTE;
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  PALETTE = {
    ink: v('--ink', '#101828'),
    inkSoft: v('--ink-soft', '#475467'),
    inkFaint: v('--ink-faint', '#98a2b3'),
    line: v('--line', '#e4e7ec'),
    lineSoft: v('--line-soft', '#f0f2f5'),
    paper: v('--paper', '#ffffff'),
    accent: v('--accent', '#1d4ed8'),
    accentSoft: v('--accent-line', '#c7d2fe'),
    accentFill: v('--accent-fill', 'rgba(29,78,216,.07)'),
    good: v('--good', '#15803d'),
    warn: v('--warn', '#b54708'),
    bad: v('--bad', '#b42318'),
    cat: [v('--cat-1', '#1d4ed8'), v('--cat-2', '#0e9384'), v('--cat-3', '#b54708'),
          v('--cat-4', '#7a5af8'), v('--cat-5', '#b42318'), v('--cat-6', '#0086c9')],
  };
  return PALETTE;
}

/** Same theme colour, softened so text stays readable on top of it. */
function withAlpha(color, alpha) {
  const hex = color.trim();
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const full = m[1].length === 3 ? m[1].split('').map((c) => c + c).join('') : m[1];
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** CSS reference for a series colour, for swatches that live in HTML. */
const catVar = (i) => `var(--cat-${(i % 6) + 1})`;
const catColor = (i) => palette().cat[i % 6];

/* --------------------------------------------------------------- layout --- */
function base(extra = {}) {
  const P = palette();
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 14;
  const size = Math.max(9, rem * 0.68);
  return Object.assign({
    margin: { l: 44, r: 12, t: 8, b: 32 },
    font: { family: FONT_FAMILY, color: P.inkSoft, size },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    hoverlabel: {
      bgcolor: P.paper, bordercolor: P.line,
      font: { family: FONT_FAMILY, color: P.ink, size },
    },
    xaxis: axis(), yaxis: axis(),
  }, extra);
}

function axis(extra = {}) {
  const P = palette();
  return Object.assign({
    gridcolor: P.line, zerolinecolor: P.line, linecolor: P.line,
    tickcolor: P.line, ticklen: 3, automargin: true,
    title: { font: { size: 10, color: P.inkFaint }, standoff: 6 },
  }, extra);
}

/* ------------------------------------------------------------- registry --- */
/**
 * Render a chart and remember how to rebuild it.
 *
 * Plotly bakes colours into the figure at draw time, so a CSS theme change does
 * not reach an existing plot. Recording the builder lets redrawAll() replay
 * every chart against the new palette instead of reloading the page.
 */
const REGISTRY = new Map();

function render(id, builder) {
  const el = document.getElementById(id);
  if (!el) return;
  REGISTRY.set(id, builder);
  const fig = builder();
  Plotly.react(el, fig.traces, fig.layout, CONFIG);
}

function redrawAll() {
  PALETTE = null;
  REGISTRY.forEach((builder, id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const fig = builder();
    Plotly.react(el, fig.traces, fig.layout, CONFIG);
  });
}

function resizeVisible(slideEl) {
  if (!slideEl) return;
  slideEl.querySelectorAll('.chart').forEach((el) => {
    if (el.data) Plotly.Plots.resize(el);
  });
}

const rupee = (v) => '₹' + Math.round(v).toLocaleString('en-IN');
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const compactRs = (v) => {
  const a = Math.abs(v);
  if (a >= 1e7) return '₹' + (v / 1e7).toFixed(2) + ' Cr';
  if (a >= 1e5) return '₹' + (v / 1e5).toFixed(1) + ' L';
  if (a >= 1e3) return '₹' + Math.round(v / 1e3) + 'k';
  return '₹' + Math.round(v);
};

/* ----------------------------------------------------- page 2 · data ---- */
function targetDonut(counts, claimRate) {
  render('chart-target', () => {
    const P = palette();
    return {
      traces: [{
        type: 'pie', hole: 0.62, sort: false,
        labels: ['No claim', 'Claim raised'],
        values: [counts.no_claim, counts.claim],
        marker: { colors: [P.line, P.accent], line: { color: P.paper, width: 2 } },
        textinfo: 'none',
        hovertemplate: '%{label}<br>%{value:,} shipments (%{percent})<extra></extra>',
      }],
      layout: base({
        margin: { l: 6, r: 6, t: 6, b: 6 },
        annotations: [{
          text: `<b style="font-size:1.5em;color:${P.ink}">${pct(claimRate)}</b><br>` +
                `<span style="color:${P.inkFaint}">claim rate</span>`,
          showarrow: false, font: { size: 12 },
        }],
      }),
    };
  });
}

function monthlyLine(monthly) {
  render('chart-monthly', () => {
    const P = palette();
    return {
      traces: [{
        type: 'scatter', mode: 'lines', x: monthly.m, y: monthly.rate,
        line: { color: P.accent, width: 2, shape: 'spline', smoothing: 0.6 },
        fill: 'tozeroy', fillcolor: P.accentFill,
        hovertemplate: '%{x}<br>claim rate %{y:.1%}<extra></extra>',
      }],
      layout: base({
        margin: { l: 40, r: 10, t: 8, b: 28 },
        xaxis: axis({ nticks: 6 }),
        yaxis: axis({ tickformat: '.0%', rangemode: 'tozero' }),
      }),
    };
  });
}

/* ------------------------------------------------------ page 3 · eda ---- */
function edaTarget(counts) {
  render('chart-eda-target', () => {
    const P = palette();
    const total = counts.no_claim + counts.claim;
    return {
      traces: [{
        type: 'bar',
        x: ['No claim', 'Claim raised'],
        y: [counts.no_claim, counts.claim],
        marker: { color: [P.line, P.accent] },
        text: [`${counts.no_claim.toLocaleString()}<br>${pct(counts.no_claim / total)}`,
               `${counts.claim.toLocaleString()}<br>${pct(counts.claim / total)}`],
        textposition: 'outside', cliponaxis: false,
        outsidetextfont: { color: P.ink, size: 13 },
        hovertemplate: '%{x}: %{y:,} shipments<extra></extra>',
      }],
      layout: base({
        margin: { l: 46, r: 16, t: 34, b: 28 }, bargap: 0.45,
        xaxis: axis({ showgrid: false }),
        yaxis: axis({ rangemode: 'tozero' }),
      }),
    };
  });
}

function edaCategory(data, baseRate) {
  render('chart-eda-cat', () => {
    const P = palette();
    return {
      traces: [{
        type: 'bar', orientation: 'h',
        y: data.levels, x: data.rate,
        marker: { color: data.rate.map((r) => (r >= baseRate ? P.accent : P.accentSoft)) },
        customdata: data.count,
        hovertemplate: '%{y}<br>claim rate %{x:.1%}<br>%{customdata:,} shipments<extra></extra>',
      }],
      layout: base({
        margin: { l: 106, r: 30, t: 8, b: 28 },
        xaxis: axis({ tickformat: '.0%', title: { text: 'claim rate' } }),
        yaxis: axis({ showgrid: false, automargin: true }),
        shapes: [{
          type: 'line', x0: baseRate, x1: baseRate,
          y0: -0.5, y1: data.levels.length - 0.5,
          line: { color: P.inkFaint, width: 1, dash: 'dot' },
        }],
        annotations: [{
          x: baseRate, y: 1.03, yref: 'paper', text: 'book average',
          showarrow: false, font: { size: 9, color: P.inkFaint }, xanchor: 'left',
        }],
      }),
    };
  });
}

function edaAmount(hist, stats) {
  render('chart-eda-amount', () => {
    const P = palette();
    const edges = hist.log10_edges;
    const centres = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    return {
      traces: [{
        type: 'bar', x: centres, y: hist.counts,
        marker: { color: P.accentSoft, line: { color: P.accent, width: 0.5 } },
        customdata: centres.map((c) => Math.pow(10, c)),
        hovertemplate: 'around ₹%{customdata:,.0f}<br>%{y} claims<extra></extra>',
      }],
      layout: base({
        margin: { l: 40, r: 14, t: 8, b: 34 }, bargap: 0.04,
        xaxis: axis({
          title: { text: 'claim amount' }, tickmode: 'array',
          tickvals: [2, 3, 4, 5, 6],
          ticktext: ['₹100', '₹1k', '₹10k', '₹1L', '₹10L'],
        }),
        yaxis: axis({ title: { text: 'claims' } }),
        shapes: [{
          type: 'line', yref: 'paper', y0: 0, y1: 1,
          x0: Math.log10(stats.median), x1: Math.log10(stats.median),
          line: { color: P.ink, width: 1.5, dash: 'dot' },
        }],
        annotations: [{
          x: Math.log10(stats.median), y: 1.02, yref: 'paper', xanchor: 'left',
          text: `median ${rupee(stats.median)}`, showarrow: false,
          font: { size: 9, color: P.inkSoft },
        }],
      }),
    };
  });
}

function edaNumeric(data, baseRate) {
  render('chart-eda-num', () => {
    const P = palette();
    return {
      traces: [{
        type: 'scatter', mode: 'lines+markers', x: data.mid, y: data.rate,
        line: { color: P.accent, width: 2.5 },
        marker: { color: P.accent, size: 7, line: { color: P.paper, width: 1.5 } },
        customdata: data.count,
        hovertemplate: '%{x}<br>claim rate %{y:.1%}<br>%{customdata:,} shipments<extra></extra>',
      }],
      layout: base({
        margin: { l: 44, r: 16, t: 10, b: 32 },
        xaxis: axis({ title: { text: data.label } }),
        yaxis: axis({ tickformat: '.0%', title: { text: 'claim rate' }, rangemode: 'tozero' }),
        shapes: [{
          type: 'line', xref: 'paper', x0: 0, x1: 1, y0: baseRate, y1: baseRate,
          line: { color: P.inkFaint, width: 1, dash: 'dot' },
        }],
      }),
    };
  });
}

/* -------------------------------------------------- page 4 · clusters --- */
function clusterScatter(points, summary) {
  render('chart-clusters', () => {
    const P = palette();
    const traces = summary.map((c) => {
      const rows = points.filter((p) => p.cluster === c.cluster);
      const volumes = rows.map((r) => r.shipments);
      const maxVol = Math.max.apply(null, volumes.concat([1]));
      return {
        type: 'scatter', mode: 'markers', name: c.archetype,
        x: rows.map((r) => r.claim_rate),
        y: rows.map((r) => r.avg_claim_value),
        text: rows.map((r) => r.lane_id),
        customdata: rows.map((r) => [r.shipments, r.expected_loss_per_shipment]),
        marker: {
          color: catColor(c.cluster), opacity: 0.72,
          size: volumes.map((v) => 6 + 9 * Math.sqrt(v / maxVol)),
          line: { color: P.paper, width: 1 },
        },
        hovertemplate: '<b>%{text}</b><br>claim rate %{x:.1%}<br>' +
                       'avg claim ₹%{y:,.0f}<br>%{customdata[0]:,} shipments<br>' +
                       'expected loss ₹%{customdata[1]:,.0f}/shipment<extra></extra>',
      };
    });

    // The fitted centres, drawn on the same axes as the lanes they summarise.
    // A bare cross disappears into 210 overlapping dots, so each one gets a
    // paper-coloured halo punched out of the cloud behind it.
    const cx = summary.map((c) => c.centroid.claim_rate);
    const cy = summary.map((c) => c.centroid.avg_claim_value);
    traces.push({
      type: 'scatter', mode: 'markers', showlegend: false, hoverinfo: 'skip',
      x: cx, y: cy,
      marker: {
        symbol: 'circle', size: 26, color: P.paper, opacity: 0.92,
        line: { width: 2, color: P.ink },
      },
    });
    traces.push({
      type: 'scatter', mode: 'markers', name: 'Centroids', showlegend: false,
      x: cx, y: cy,
      text: summary.map((c) => c.archetype),
      marker: {
        symbol: 'x-thin', size: 15,
        line: { width: 4, color: summary.map((c) => catColor(c.cluster)) },
      },
      hovertemplate: '<b>%{text}</b><br>cluster centre<br>' +
                     'claim rate %{x:.1%} · avg claim ₹%{y:,.0f}<extra></extra>',
    });

    return {
      traces,
      layout: base({
        margin: { l: 58, r: 16, t: 22, b: 40 },
        showlegend: true,
        legend: {
          orientation: 'h', y: 1.14, x: 0, font: { size: 9.5 },
          bgcolor: 'rgba(0,0,0,0)', itemsizing: 'constant',
        },
        xaxis: axis({ tickformat: '.0%', title: { text: 'historical claim rate' } }),
        yaxis: axis({ type: 'log', dtick: 1, tickprefix: '₹',
                      title: { text: 'average claim value (log)' } }),
      }),
    };
  });
}

/**
 * What actually separates the clusters.
 *
 * The scatter can only show 2 of the 9 features K-Means used, which is why the
 * colours look intermingled. This shows all nine at once as standardised
 * deviations from the average lane, so the overlap stops being a mystery.
 *
 * NOT CURRENTLY ON THE DECK. Removed from slide 4 because it needs more
 * explaining than it earns in a live presentation. Everything it needs is still
 * computed and served (`profile_z` per cluster), so putting it back is two
 * steps: add `<div id="chart-cluster-profile" class="chart">` inside a card on
 * slide 4, and call this from refreshClusters() in presentation.js.
 */
function clusterHeatmap(summary, features, labels) {
  render('chart-cluster-profile', () => {
    const P = palette();
    const z = features.map((f) => summary.map((c) => c.profile_z[f]));
    const bound = Math.max(1, ...z.flat().map(Math.abs));
    return {
      traces: [{
        type: 'heatmap',
        z,
        x: summary.map((c) => c.archetype),
        y: features.map((f) => labels[f] || f),
        zmid: 0, zmin: -bound, zmax: bound,
        // Tints rather than solid fills: only the genuinely distinguishing cells
        // carry colour, and every cell stays light enough for its number to read
        // in either theme. Full-strength fills would win the contrast fight.
        colorscale: [
          [0.0, withAlpha(P.warn, 0.55)], [0.30, withAlpha(P.warn, 0.20)],
          [0.5, withAlpha(P.inkFaint, 0.07)],
          [0.70, withAlpha(P.accent, 0.20)], [1.0, withAlpha(P.accent, 0.55)],
        ],
        showscale: false, xgap: 3, ygap: 3,
        text: z.map((row) => row.map(
          (v) => (Math.abs(v) < 0.15 ? '' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1)))),
        texttemplate: '%{text}',
        textfont: { size: 10, color: P.ink },
        hovertemplate: '%{x}<br>%{y}: %{z:+.2f} SD from the average lane<extra></extra>',
      }],
      layout: base({
        margin: { l: 138, r: 8, t: 6, b: 34 },
        xaxis: axis({ showgrid: false, side: 'bottom', tickfont: { size: 9 },
                      tickangle: 0, automargin: true }),
        yaxis: axis({ showgrid: false, autorange: 'reversed', tickfont: { size: 9.5 } }),
      }),
    };
  });
}

function elbowChart(elbow, currentK) {
  render('chart-elbow', () => {
    const P = palette();
    return {
      traces: [
        {
          type: 'scatter', mode: 'lines+markers', name: 'Inertia',
          x: elbow.map((e) => e.k), y: elbow.map((e) => e.inertia),
          line: { color: P.accent, width: 2 },
          marker: { size: elbow.map((e) => (e.k === currentK ? 11 : 6)), color: P.accent },
          hovertemplate: 'K=%{x}<br>inertia %{y:.0f}<extra></extra>',
        },
        {
          type: 'scatter', mode: 'lines+markers', name: 'Silhouette', yaxis: 'y2',
          x: elbow.map((e) => e.k), y: elbow.map((e) => e.silhouette),
          line: { color: P.warn, width: 2, dash: 'dot' },
          marker: { size: elbow.map((e) => (e.k === currentK ? 11 : 6)), color: P.warn },
          hovertemplate: 'K=%{x}<br>silhouette %{y:.3f}<extra></extra>',
        },
      ],
      layout: base({
        margin: { l: 30, r: 30, t: 10, b: 30 },
        xaxis: axis({ dtick: 1, title: { text: 'K' } }),
        yaxis: axis({ showticklabels: false, title: { text: 'inertia' } }),
        yaxis2: axis({ overlaying: 'y', side: 'right', showgrid: false,
                       showticklabels: false, title: { text: 'silhouette' } }),
      }),
    };
  });
}

/* ----------------------------------------------- page 5 · probability --- */
function probDistribution(hist, marker, markerLabel) {
  render('chart-prob-dist', () => {
    const P = palette();
    const edges = hist.edges;
    const centres = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    return {
      traces: [{
        type: 'bar', x: centres, y: hist.counts,
        marker: { color: centres.map((c) => (c <= marker ? P.accentSoft : P.line)) },
        hovertemplate: 'p ≈ %{x:.2f}<br>%{y:,} test shipments<extra></extra>',
      }],
      layout: base({
        margin: { l: 44, r: 14, t: 30, b: 34 }, bargap: 0.05,
        xaxis: axis({ title: { text: 'predicted claim probability' }, tickformat: '.0%' }),
        yaxis: axis({ title: { text: 'test shipments' } }),
        shapes: [{
          type: 'line', yref: 'paper', y0: 0, y1: 1, x0: marker, x1: marker,
          line: { color: P.bad, width: 2 },
        }],
        annotations: [{
          x: marker, y: 1.06, yref: 'paper', text: markerLabel, showarrow: false,
          font: { size: 10, color: P.bad }, xanchor: 'center',
        }],
      }),
    };
  });
}

/* ------------------------------------------------- page 6 · threshold --- */
function rocChart(id, curve, point) {
  render(id, () => {
    const P = palette();
    return {
      traces: [
        { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], showlegend: false,
          line: { color: P.line, width: 1, dash: 'dash' }, hoverinfo: 'skip' },
        { type: 'scatter', mode: 'lines', x: curve.roc.fpr, y: curve.roc.tpr,
          line: { color: P.accent, width: 2.5 }, fill: 'tozeroy',
          fillcolor: P.accentFill,
          hovertemplate: 'FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>' },
        { type: 'scatter', mode: 'markers', x: [point.fpr], y: [point.tpr],
          marker: { color: P.bad, size: 15, line: { color: P.paper, width: 2.5 } },
          hovertemplate: `threshold ${point.threshold.toFixed(2)}<br>` +
                         'FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>' },
      ],
      layout: base({
        margin: { l: 42, r: 14, t: 20, b: 34 },
        xaxis: axis({ title: { text: 'false positive rate' }, range: [-0.02, 1.02] }),
        yaxis: axis({ title: { text: 'true positive rate' }, range: [-0.02, 1.02] }),
        annotations: [{
          x: 0.97, y: 0.06, xanchor: 'right', text: `AUC ${curve.roc_auc.toFixed(3)}`,
          showarrow: false, font: { size: 11, color: P.accent },
        }],
      }),
    };
  });
}

function prChart(id, curve, point, baseRate) {
  render(id, () => {
    const P = palette();
    return {
      traces: [
        { type: 'scatter', mode: 'lines', x: [0, 1], y: [baseRate, baseRate],
          showlegend: false, line: { color: P.line, width: 1, dash: 'dash' },
          hoverinfo: 'skip' },
        { type: 'scatter', mode: 'lines', x: curve.pr.recall, y: curve.pr.precision,
          line: { color: P.accent, width: 2.5 },
          hovertemplate: 'recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>' },
        { type: 'scatter', mode: 'markers', x: [point.recall], y: [point.precision],
          marker: { color: P.bad, size: 15, line: { color: P.paper, width: 2.5 } },
          hovertemplate: `threshold ${point.threshold.toFixed(2)}<br>` +
                         'recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>' },
      ],
      layout: base({
        margin: { l: 42, r: 14, t: 20, b: 34 },
        xaxis: axis({ title: { text: 'recall' }, range: [-0.02, 1.02] }),
        yaxis: axis({ title: { text: 'precision' }, range: [0, 1] }),
        annotations: [
          { x: 0.97, y: 0.94, xanchor: 'right',
            text: `PR-AUC ${curve.pr_auc.toFixed(3)}`, showarrow: false,
            font: { size: 11, color: P.accent } },
          { x: 0.02, y: baseRate, xanchor: 'left', yanchor: 'bottom', text: 'no-skill',
            showarrow: false, font: { size: 9, color: P.inkFaint } },
        ],
      }),
    };
  });
}

/* -------------------------------------------------- page 7 · trade-off -- */
function tradeoffChart(sweep, threshold) {
  render('chart-tradeoff', () => {
    const P = palette();
    const series = [
      ['precision', 'Hit rate (precision)', P.accent],
      ['recall', 'Claims caught (recall)', P.good],
      ['f1', 'F1', P.warn],
    ];
    return {
      traces: series.map(([key, name, color]) => ({
        type: 'scatter', mode: 'lines', name,
        x: sweep.threshold, y: sweep[key],
        line: { color, width: 2.2 }, connectgaps: false,
        hovertemplate: `${name} %{y:.3f} at threshold %{x:.2f}<extra></extra>`,
      })),
      layout: base({
        margin: { l: 40, r: 14, t: 24, b: 34 },
        showlegend: true,
        legend: { orientation: 'h', y: 1.16, x: 0, font: { size: 9.5 } },
        xaxis: axis({ title: { text: 'decision threshold' }, range: [0, 1] }),
        yaxis: axis({ range: [0, 1] }),
        shapes: [
          { type: 'line', yref: 'paper', y0: 0, y1: 1,
            x0: sweep.best_f1_threshold, x1: sweep.best_f1_threshold,
            line: { color: P.inkFaint, width: 1, dash: 'dot' } },
          { type: 'line', yref: 'paper', y0: 0, y1: 1, x0: threshold, x1: threshold,
            line: { color: P.bad, width: 2 } },
        ],
        annotations: [
          { x: sweep.best_f1_threshold, y: 0.02, yref: 'paper', xanchor: 'left',
            text: `best F1 ${sweep.best_f1_threshold.toFixed(2)}`, showarrow: false,
            font: { size: 9, color: P.inkFaint } },
          { x: threshold, y: 1.04, yref: 'paper', text: threshold.toFixed(2),
            showarrow: false, font: { size: 10, color: P.bad } },
        ],
      }),
    };
  });
}

/* ------------------------------------------------ page 8 · cost curve --- */
/**
 * Total cost of every threshold, with the cheapest one marked.
 *
 * The do-nothing line matters as much as the curve: flagging nothing is always
 * an option, and once an intervention costs enough it becomes the right one.
 */
function costCurve(data) {
  render('chart-cost', () => {
    const P = palette();
    const c = data.curve;
    return {
      traces: [
        // No fill: 'tozeroy' would drag the axis to zero, and since every policy
        // costs at least Rs 1.2 crore that flattens the whole curve into a line.
        { type: 'scatter', mode: 'lines', x: c.thresholds, y: c.costs,
          line: { color: P.accent, width: 2.5 }, name: 'Total cost',
          hovertemplate: 'threshold %{x:.2f}<br>total cost ₹%{y:,.0f}<extra></extra>' },
        { type: 'scatter', mode: 'markers', x: [c.optimal_threshold], y: [c.optimal_cost],
          marker: { color: P.good, size: 16, symbol: 'circle',
                    line: { color: P.paper, width: 2.5 } },
          name: 'Cheapest',
          hovertemplate: `cheapest at ${c.optimal_threshold.toFixed(2)}` +
                         '<br>₹%{y:,.0f}<extra></extra>' },
        { type: 'scatter', mode: 'markers', x: [data.current_threshold], y: [data.current_cost],
          marker: { color: P.bad, size: 13, symbol: 'diamond',
                    line: { color: P.paper, width: 2 } },
          name: 'Current',
          hovertemplate: `current ${data.current_threshold.toFixed(2)}` +
                         '<br>₹%{y:,.0f}<extra></extra>' },
      ],
      layout: base({
        margin: { l: 56, r: 14, t: 26, b: 36 },
        showlegend: true,
        legend: { orientation: 'h', y: 1.18, x: 0, font: { size: 9.5 } },
        xaxis: axis({ title: { text: 'decision threshold' }, range: [0, 1] }),
        // Deliberately not anchored at zero: every policy costs at least Rs 1.2 Cr,
        // and a zero baseline would compress the differences that matter into a
        // flat line. The do-nothing reference line supplies the anchor instead.
        yaxis: axis({ title: { text: 'total cost over the test set' },
                      tickformat: '.3s', tickprefix: '₹' }),
        shapes: [{
          type: 'line', xref: 'paper', x0: 0, x1: 1,
          y0: c.do_nothing_cost, y1: c.do_nothing_cost,
          line: { color: P.inkFaint, width: 1, dash: 'dash' },
        }],
        annotations: [{
          x: 0.02, y: c.do_nothing_cost, xanchor: 'left', yanchor: 'bottom',
          text: `do nothing · ${compactRs(c.do_nothing_cost)}`,
          showarrow: false, font: { size: 9, color: P.inkFaint },
        }],
      }),
    };
  });
}

/**
 * What each policy saves against doing nothing.
 *
 * Plotting the three total costs side by side would hide the story: they all sit
 * between Rs 1.2 and 1.4 crore, so a 12% gap looks like no gap at all. The
 * saving is the decision-relevant number, it starts honestly at zero, and a
 * policy that costs more than inaction shows up as a bar below the line.
 */
function costComparison(data) {
  render('chart-cost-compare', () => {
    const P = palette();
    const c = data.curve;
    const rows = [
      ['Current threshold', c.do_nothing_cost - data.current_cost, P.bad],
      ['Cheapest threshold', c.do_nothing_cost - c.optimal_cost, P.accent],
      ['Rank by expected loss',
       c.do_nothing_cost - data.expected_loss_rule.cost, P.good],
    ];
    return {
      traces: [{
        type: 'bar', orientation: 'h',
        y: rows.map((r) => r[0]), x: rows.map((r) => r[1]),
        marker: { color: rows.map((r) => r[2]) },
        text: rows.map((r) => (r[1] >= 0 ? compactRs(r[1]) : '−' + compactRs(-r[1]))),
        textposition: 'outside', cliponaxis: false,
        outsidetextfont: { color: P.ink, size: 11 },
        hovertemplate: '%{y}<br>saves ₹%{x:,.0f} vs doing nothing<extra></extra>',
      }],
      layout: base({
        margin: { l: 124, r: 58, t: 10, b: 32 }, bargap: 0.4,
        xaxis: (() => {
          const values = rows.map((r) => r[1]);
          const span = Math.max.apply(null, values.map(Math.abs)) || 1;
          return axis({
            title: { text: 'saved vs doing nothing' },
            tickformat: '.2s', tickprefix: '₹',
            // Headroom on both sides so an outside label never lands on the
            // category names, including when a policy saves less than nothing.
            range: [Math.min(0, Math.min.apply(null, values)) - span * 0.16,
                    Math.max.apply(null, values) + span * 0.16],
            zeroline: true, zerolinecolor: P.inkFaint, zerolinewidth: 1.5,
          });
        })(),
        yaxis: axis({ showgrid: false, autorange: 'reversed' }),
      }),
    };
  });
}

/* ------------------------------------------------ page 9 · claim value -- */
function actualVsPredicted(scatter) {
  render('chart-avp', () => {
    const P = palette();
    const all = scatter.actual.concat(scatter.predicted).filter((v) => v > 0);
    const lo = Math.max(10, Math.min.apply(null, all));
    const hi = Math.max.apply(null, all);
    return {
      traces: [
        { type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi],
          line: { color: P.inkFaint, width: 1, dash: 'dash' }, hoverinfo: 'skip' },
        { type: 'scatter', mode: 'markers', x: scatter.predicted, y: scatter.actual,
          marker: { color: P.accent, size: 5, opacity: 0.42 },
          hovertemplate: 'predicted ₹%{x:,.0f}<br>actual ₹%{y:,.0f}<extra></extra>' },
      ],
      layout: base({
        margin: { l: 50, r: 14, t: 10, b: 36 },
        xaxis: axis({ type: 'log', dtick: 1, tickprefix: '₹',
                      title: { text: 'predicted (log)' } }),
        yaxis: axis({ type: 'log', dtick: 1, tickprefix: '₹',
                      title: { text: 'actual (log)' } }),
      }),
    };
  });
}

function residualPlot(scatter) {
  render('chart-resid', () => {
    const P = palette();
    return {
      traces: [{
        type: 'scatter', mode: 'markers',
        x: scatter.predicted, y: scatter.residual,
        marker: { color: scatter.residual.map((r) => (r >= 0 ? P.accent : P.warn)),
                  size: 5, opacity: 0.45 },
        hovertemplate: 'predicted ₹%{x:,.0f}<br>residual ₹%{y:,.0f}<extra></extra>',
      }],
      layout: base({
        margin: { l: 52, r: 14, t: 10, b: 36 },
        xaxis: axis({ title: { text: 'predicted claim value' }, tickprefix: '₹' }),
        yaxis: axis({ title: { text: 'actual − predicted' }, tickprefix: '₹',
                      zeroline: true, zerolinecolor: P.inkFaint, zerolinewidth: 1.5 }),
      }),
    };
  });
}

/* ------------------------------------------- page 10 · model comparison - */
function comparisonCurves(models, order) {
  render('chart-cmp-roc', () => {
    const P = palette();
    const traces = order.map((key, i) => ({
      type: 'scatter', mode: 'lines', name: models[key].name,
      x: models[key].curves.roc.fpr, y: models[key].curves.roc.tpr,
      line: { color: catColor(i), width: 2.2 },
      hovertemplate: `${models[key].name}<br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>`,
    }));
    traces.unshift({
      type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], showlegend: false,
      line: { color: P.line, width: 1, dash: 'dash' }, hoverinfo: 'skip',
    });
    return {
      traces,
      layout: base({
        margin: { l: 42, r: 14, t: 26, b: 34 },
        showlegend: true, legend: { orientation: 'h', y: 1.2, x: 0, font: { size: 9 } },
        xaxis: axis({ title: { text: 'false positive rate' } }),
        yaxis: axis({ title: { text: 'true positive rate' } }),
      }),
    };
  });

  render('chart-cmp-pr', () => {
    const P = palette();
    const baseRate = models[order[0]].curves.positive_rate;
    const traces = order.map((key, i) => ({
      type: 'scatter', mode: 'lines', name: models[key].name,
      x: models[key].curves.pr.recall, y: models[key].curves.pr.precision,
      line: { color: catColor(i), width: 2.2 },
      hovertemplate: `${models[key].name}<br>recall %{x:.3f} · precision %{y:.3f}<extra></extra>`,
    }));
    traces.unshift({
      type: 'scatter', mode: 'lines', x: [0, 1], y: [baseRate, baseRate],
      showlegend: false, line: { color: P.line, width: 1, dash: 'dash' },
      hoverinfo: 'skip',
    });
    return {
      traces,
      layout: base({
        margin: { l: 42, r: 14, t: 26, b: 34 },
        showlegend: true, legend: { orientation: 'h', y: 1.2, x: 0, font: { size: 9 } },
        xaxis: axis({ title: { text: 'recall' } }),
        yaxis: axis({ title: { text: 'precision' }, range: [0, 1] }),
      }),
    };
  });
}

window.Charts = {
  catVar, catColor, rupee, pct, compactRs, resizeVisible, redrawAll,
  targetDonut, monthlyLine,
  edaTarget, edaCategory, edaAmount, edaNumeric,
  clusterScatter, clusterHeatmap, elbowChart,
  probDistribution, rocChart, prChart, tradeoffChart,
  costCurve, costComparison,
  actualVsPredicted, residualPlot, comparisonCurves,
};
