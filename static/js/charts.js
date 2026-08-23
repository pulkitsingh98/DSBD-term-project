/* ==========================================================================
   Chart layer.

   One Plotly theme for the whole deck: light ground, hairline axes, a single
   accent, and a categorical palette used only where categories genuinely need
   separating (clusters, model comparison). Every figure answers one question,
   so titles live in the surrounding card and never inside the plot.
   ========================================================================== */

const INK = '#101828', INK_SOFT = '#475467', INK_FAINT = '#98a2b3';
const LINE = '#e4e7ec', ACCENT = '#1d4ed8', ACCENT_SOFT = '#c7d2fe';
const GOOD = '#15803d', WARN = '#b54708', BAD = '#b42318';
const CAT = ['#1d4ed8', '#0e9384', '#b54708', '#7a5af8', '#b42318', '#0086c9'];

const FONT = {
  family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  color: INK_SOFT,
};

const CONFIG = { displayModeBar: false, responsive: true, doubleClick: 'reset' };

/** Base layout every chart starts from. */
function base(extra = {}) {
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 14;
  return Object.assign({
    margin: { l: 44, r: 12, t: 8, b: 32 },
    font: Object.assign({}, FONT, { size: Math.max(9, rem * 0.72) }),
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    hoverlabel: {
      bgcolor: '#fff', bordercolor: LINE,
      font: Object.assign({}, FONT, { color: INK, size: Math.max(9, rem * 0.72) }),
    },
    xaxis: axis(), yaxis: axis(),
  }, extra);
}

function axis(extra = {}) {
  return Object.assign({
    gridcolor: LINE, zerolinecolor: LINE, linecolor: LINE,
    tickcolor: LINE, ticklen: 3, automargin: true,
    title: { font: { size: 10, color: INK_FAINT }, standoff: 6 },
  }, extra);
}

const rupee = (v) => '₹' + Math.round(v).toLocaleString('en-IN');
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';

function draw(id, traces, layout) {
  const el = document.getElementById(id);
  if (!el) return;
  Plotly.react(el, traces, layout, CONFIG);
}

/** Re-fit every chart on the visible slide (called on resize / slide change). */
function resizeVisible(slideEl) {
  if (!slideEl) return;
  slideEl.querySelectorAll('.chart').forEach((el) => {
    if (el.data) Plotly.Plots.resize(el);
  });
}

/* ----------------------------------------------------- page 2 · data ---- */
function targetDonut(counts, claimRate) {
  draw('chart-target', [{
    type: 'pie', hole: 0.62, sort: false,
    labels: ['No claim', 'Claim raised'],
    values: [counts.no_claim, counts.claim],
    marker: { colors: [LINE, ACCENT], line: { color: '#fff', width: 2 } },
    textinfo: 'none',
    hovertemplate: '%{label}<br>%{value:,} shipments (%{percent})<extra></extra>',
  }], base({
    margin: { l: 6, r: 6, t: 6, b: 6 },
    annotations: [{
      text: `<b style="font-size:1.5em;color:${INK}">${pct(claimRate)}</b><br>` +
            `<span style="color:${INK_FAINT}">claim rate</span>`,
      showarrow: false, font: { size: 12 },
    }],
  }));
}

function monthlyLine(monthly) {
  draw('chart-monthly', [{
    type: 'scatter', mode: 'lines', x: monthly.m, y: monthly.rate,
    line: { color: ACCENT, width: 2, shape: 'spline', smoothing: 0.6 },
    fill: 'tozeroy', fillcolor: 'rgba(29,78,216,.07)',
    hovertemplate: '%{x}<br>claim rate %{y:.1%}<extra></extra>',
  }], base({
    margin: { l: 40, r: 10, t: 8, b: 28 },
    xaxis: axis({ nticks: 6 }),
    yaxis: axis({ tickformat: '.0%', rangemode: 'tozero' }),
  }));
}

/* ------------------------------------------------------ page 3 · eda ---- */
function edaTarget(counts) {
  const total = counts.no_claim + counts.claim;
  draw('chart-eda-target', [{
    type: 'bar',
    x: ['No claim', 'Claim raised'],
    y: [counts.no_claim, counts.claim],
    marker: { color: [LINE, ACCENT] },
    text: [`${counts.no_claim.toLocaleString()}<br>${pct(counts.no_claim / total)}`,
           `${counts.claim.toLocaleString()}<br>${pct(counts.claim / total)}`],
    textposition: 'outside', cliponaxis: false,
    outsidetextfont: { color: INK, size: 13 },
    hovertemplate: '%{x}: %{y:,} shipments<extra></extra>',
  }], base({
    margin: { l: 46, r: 16, t: 34, b: 28 }, bargap: 0.45,
    xaxis: axis({ showgrid: false }),
    yaxis: axis({ showgrid: true, rangemode: 'tozero' }),
  }));
}

function edaCategory(data, baseRate) {
  draw('chart-eda-cat', [{
    type: 'bar', orientation: 'h',
    y: data.levels, x: data.rate,
    marker: {
      color: data.rate.map((r) => (r >= baseRate ? ACCENT : ACCENT_SOFT)),
    },
    customdata: data.count,
    hovertemplate: '%{y}<br>claim rate %{x:.1%}<br>%{customdata:,} shipments<extra></extra>',
  }], base({
    margin: { l: 106, r: 30, t: 8, b: 28 },
    xaxis: axis({ tickformat: '.0%', title: { text: 'claim rate' } }),
    yaxis: axis({ showgrid: false, automargin: true }),
    shapes: [{
      type: 'line', x0: baseRate, x1: baseRate, y0: -0.5, y1: data.levels.length - 0.5,
      line: { color: INK_FAINT, width: 1, dash: 'dot' },
    }],
    annotations: [{
      x: baseRate, y: 1.03, yref: 'paper', text: 'book average',
      showarrow: false, font: { size: 9, color: INK_FAINT }, xanchor: 'left',
    }],
  }));
}

function edaAmount(hist, stats) {
  const edges = hist.log10_edges;
  const centres = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  draw('chart-eda-amount', [{
    type: 'bar', x: centres, y: hist.counts,
    marker: { color: ACCENT_SOFT, line: { color: ACCENT, width: 0.5 } },
    customdata: centres.map((c) => Math.pow(10, c)),
    hovertemplate: 'around ₹%{customdata:,.0f}<br>%{y} claims<extra></extra>',
  }], base({
    margin: { l: 40, r: 14, t: 8, b: 34 },
    bargap: 0.04,
    xaxis: axis({
      title: { text: 'claim amount' },
      tickmode: 'array',
      tickvals: [2, 3, 4, 5, 6],
      ticktext: ['₹100', '₹1k', '₹10k', '₹1L', '₹10L'],
    }),
    yaxis: axis({ title: { text: 'claims' } }),
    shapes: [{
      type: 'line', yref: 'paper', y0: 0, y1: 1,
      x0: Math.log10(stats.median), x1: Math.log10(stats.median),
      line: { color: INK, width: 1.5, dash: 'dot' },
    }],
    annotations: [{
      x: Math.log10(stats.median), y: 1.02, yref: 'paper', xanchor: 'left',
      text: `median ${rupee(stats.median)}`, showarrow: false,
      font: { size: 9, color: INK_SOFT },
    }],
  }));
}

function edaNumeric(data, baseRate) {
  draw('chart-eda-num', [{
    type: 'scatter', mode: 'lines+markers', x: data.mid, y: data.rate,
    line: { color: ACCENT, width: 2.5 },
    marker: { color: ACCENT, size: 7, line: { color: '#fff', width: 1.5 } },
    customdata: data.count,
    hovertemplate: '%{x}<br>claim rate %{y:.1%}<br>%{customdata:,} shipments<extra></extra>',
  }], base({
    margin: { l: 44, r: 16, t: 10, b: 32 },
    xaxis: axis({ title: { text: data.label } }),
    yaxis: axis({ tickformat: '.0%', title: { text: 'claim rate' }, rangemode: 'tozero' }),
    shapes: [{
      type: 'line', xref: 'paper', x0: 0, x1: 1, y0: baseRate, y1: baseRate,
      line: { color: INK_FAINT, width: 1, dash: 'dot' },
    }],
  }));
}

/* -------------------------------------------------- page 4 · clusters --- */
function clusterScatter(points, summary) {
  const traces = summary.map((c) => {
    const rows = points.filter((p) => p.cluster === c.cluster);
    return {
      type: 'scatter', mode: 'markers', name: c.archetype,
      x: rows.map((r) => r.claim_rate),
      y: rows.map((r) => r.avg_claim_value),
      text: rows.map((r) => r.lane_id),
      customdata: rows.map((r) => [r.shipments, r.expected_loss_per_shipment]),
      marker: {
        color: CAT[c.cluster % CAT.length], size: 9, opacity: 0.78,
        line: { color: '#fff', width: 1 },
      },
      hovertemplate: '<b>%{text}</b><br>claim rate %{x:.1%}<br>' +
                     'avg claim ₹%{y:,.0f}<br>%{customdata[0]:,} shipments<br>' +
                     'expected loss ₹%{customdata[1]:,.0f}/shipment<extra></extra>',
    };
  });
  draw('chart-clusters', traces, base({
    margin: { l: 58, r: 16, t: 22, b: 40 },
    showlegend: true,
    legend: {
      orientation: 'h', y: 1.14, x: 0, font: { size: 9.5 },
      bgcolor: 'rgba(0,0,0,0)', itemsizing: 'constant',
    },
    xaxis: axis({ tickformat: '.0%', title: { text: 'historical claim rate' } }),
    yaxis: axis({ type: 'log', dtick: 1, title: { text: 'average claim value (log)' },
                  tickprefix: '₹' }),
  }));
}

function elbowChart(elbow, currentK) {
  draw('chart-elbow', [
    {
      type: 'scatter', mode: 'lines+markers', name: 'Inertia',
      x: elbow.map((e) => e.k), y: elbow.map((e) => e.inertia),
      line: { color: ACCENT, width: 2 },
      marker: { size: elbow.map((e) => (e.k === currentK ? 11 : 6)), color: ACCENT },
      hovertemplate: 'K=%{x}<br>inertia %{y:.0f}<extra></extra>',
    },
    {
      type: 'scatter', mode: 'lines+markers', name: 'Silhouette', yaxis: 'y2',
      x: elbow.map((e) => e.k), y: elbow.map((e) => e.silhouette),
      line: { color: WARN, width: 2, dash: 'dot' },
      marker: { size: elbow.map((e) => (e.k === currentK ? 11 : 6)), color: WARN },
      hovertemplate: 'K=%{x}<br>silhouette %{y:.3f}<extra></extra>',
    },
  ], base({
    margin: { l: 30, r: 30, t: 10, b: 30 },
    xaxis: axis({ dtick: 1, title: { text: 'K' } }),
    yaxis: axis({ showticklabels: false, title: { text: 'inertia' } }),
    yaxis2: axis({
      overlaying: 'y', side: 'right', showgrid: false,
      showticklabels: false, title: { text: 'silhouette' },
    }),
  }));
}

/* ----------------------------------------------- page 5 · probability --- */
function probDistribution(hist, marker, markerLabel) {
  const edges = hist.edges;
  const centres = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  draw('chart-prob-dist', [{
    type: 'bar', x: centres, y: hist.counts,
    marker: { color: centres.map((c) => (c <= marker ? ACCENT_SOFT : LINE)) },
    hovertemplate: 'p ≈ %{x:.2f}<br>%{y:,} test shipments<extra></extra>',
  }], base({
    margin: { l: 44, r: 14, t: 30, b: 34 }, bargap: 0.05,
    xaxis: axis({ title: { text: 'predicted claim probability' }, tickformat: '.0%' }),
    yaxis: axis({ title: { text: 'test shipments' } }),
    shapes: [{
      type: 'line', yref: 'paper', y0: 0, y1: 1, x0: marker, x1: marker,
      line: { color: BAD, width: 2 },
    }],
    annotations: [{
      x: marker, y: 1.06, yref: 'paper', text: markerLabel, showarrow: false,
      font: { size: 10, color: BAD }, xanchor: 'center',
    }],
  }));
}

/* ------------------------------------------------- page 6 · threshold --- */
function rocChart(id, curve, point) {
  draw(id, [
    {
      type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1],
      line: { color: LINE, width: 1, dash: 'dash' }, hoverinfo: 'skip',
    },
    {
      type: 'scatter', mode: 'lines', x: curve.roc.fpr, y: curve.roc.tpr,
      line: { color: ACCENT, width: 2.5 }, fill: 'tozeroy',
      fillcolor: 'rgba(29,78,216,.07)',
      hovertemplate: 'FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>',
    },
    {
      type: 'scatter', mode: 'markers', x: [point.fpr], y: [point.tpr],
      marker: { color: BAD, size: 15, line: { color: '#fff', width: 2.5 } },
      hovertemplate: `threshold ${point.threshold.toFixed(2)}<br>` +
                     'FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>',
    },
  ], base({
    margin: { l: 42, r: 14, t: 20, b: 34 },
    xaxis: axis({ title: { text: 'false positive rate' }, range: [-0.02, 1.02] }),
    yaxis: axis({ title: { text: 'true positive rate' }, range: [-0.02, 1.02] }),
    annotations: [{
      x: 0.97, y: 0.06, xanchor: 'right', text: `AUC ${curve.roc_auc.toFixed(3)}`,
      showarrow: false, font: { size: 11, color: ACCENT },
    }],
  }));
}

function prChart(id, curve, point, baseRate) {
  draw(id, [
    {
      type: 'scatter', mode: 'lines', x: [0, 1], y: [baseRate, baseRate],
      line: { color: LINE, width: 1, dash: 'dash' }, hoverinfo: 'skip',
    },
    {
      type: 'scatter', mode: 'lines', x: curve.pr.recall, y: curve.pr.precision,
      line: { color: ACCENT, width: 2.5 },
      hovertemplate: 'recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>',
    },
    {
      type: 'scatter', mode: 'markers', x: [point.recall], y: [point.precision],
      marker: { color: BAD, size: 15, line: { color: '#fff', width: 2.5 } },
      hovertemplate: `threshold ${point.threshold.toFixed(2)}<br>` +
                     'recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>',
    },
  ], base({
    margin: { l: 42, r: 14, t: 20, b: 34 },
    xaxis: axis({ title: { text: 'recall' }, range: [-0.02, 1.02] }),
    yaxis: axis({ title: { text: 'precision' }, range: [0, 1] }),
    annotations: [{
      x: 0.97, y: 0.94, xanchor: 'right', text: `PR-AUC ${curve.pr_auc.toFixed(3)}`,
      showarrow: false, font: { size: 11, color: ACCENT },
    }, {
      x: 0.02, y: baseRate, xanchor: 'left', yanchor: 'bottom',
      text: 'no-skill', showarrow: false, font: { size: 9, color: INK_FAINT },
    }],
  }));
}

/* -------------------------------------------------- page 7 · trade-off -- */
function tradeoffChart(sweep, threshold) {
  const series = [
    ['precision', 'Precision', ACCENT],
    ['recall', 'Recall', GOOD],
    ['f1', 'F1', WARN],
  ];
  const traces = series.map(([key, name, color]) => ({
    type: 'scatter', mode: 'lines', name,
    x: sweep.threshold, y: sweep[key],
    line: { color, width: 2.2 }, connectgaps: false,
    hovertemplate: `${name} %{y:.3f} at threshold %{x:.2f}<extra></extra>`,
  }));
  draw('chart-tradeoff', traces, base({
    margin: { l: 40, r: 14, t: 24, b: 34 },
    showlegend: true,
    legend: { orientation: 'h', y: 1.16, x: 0, font: { size: 9.5 } },
    xaxis: axis({ title: { text: 'decision threshold' }, range: [0, 1] }),
    yaxis: axis({ range: [0, 1] }),
    shapes: [
      {
        type: 'line', yref: 'paper', y0: 0, y1: 1,
        x0: sweep.best_f1_threshold, x1: sweep.best_f1_threshold,
        line: { color: INK_FAINT, width: 1, dash: 'dot' },
      },
      {
        type: 'line', yref: 'paper', y0: 0, y1: 1, x0: threshold, x1: threshold,
        line: { color: BAD, width: 2 },
      },
    ],
    annotations: [
      {
        x: sweep.best_f1_threshold, y: 0.02, yref: 'paper', xanchor: 'left',
        text: `best F1 ${sweep.best_f1_threshold.toFixed(2)}`,
        showarrow: false, font: { size: 9, color: INK_FAINT },
      },
      {
        x: threshold, y: 1.04, yref: 'paper', text: threshold.toFixed(2),
        showarrow: false, font: { size: 10, color: BAD },
      },
    ],
  }));
}

/* ------------------------------------------------ page 8 · claim value -- */
function actualVsPredicted(scatter) {
  const all = scatter.actual.concat(scatter.predicted).filter((v) => v > 0);
  const lo = Math.max(10, Math.min.apply(null, all));
  const hi = Math.max.apply(null, all);
  draw('chart-avp', [
    {
      type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi],
      line: { color: INK_FAINT, width: 1, dash: 'dash' }, hoverinfo: 'skip',
    },
    {
      type: 'scatter', mode: 'markers',
      x: scatter.predicted, y: scatter.actual,
      marker: { color: ACCENT, size: 5, opacity: 0.4,
                line: { color: 'rgba(255,255,255,.5)', width: 0.5 } },
      hovertemplate: 'predicted ₹%{x:,.0f}<br>actual ₹%{y:,.0f}<extra></extra>',
    },
  ], base({
    margin: { l: 50, r: 14, t: 10, b: 36 },
    xaxis: axis({ type: 'log', dtick: 1, title: { text: 'predicted (log)' }, tickprefix: '₹' }),
    yaxis: axis({ type: 'log', dtick: 1, title: { text: 'actual (log)' }, tickprefix: '₹' }),
  }));
}

function residualPlot(scatter) {
  draw('chart-resid', [
    {
      type: 'scatter', mode: 'markers',
      x: scatter.predicted, y: scatter.residual,
      marker: { color: scatter.residual.map((r) => (r >= 0 ? ACCENT : WARN)),
                size: 5, opacity: 0.42 },
      hovertemplate: 'predicted ₹%{x:,.0f}<br>residual ₹%{y:,.0f}<extra></extra>',
    },
  ], base({
    margin: { l: 52, r: 14, t: 10, b: 36 },
    xaxis: axis({ title: { text: 'predicted claim value' }, tickprefix: '₹' }),
    yaxis: axis({ title: { text: 'actual − predicted' }, tickprefix: '₹',
                  zeroline: true, zerolinecolor: INK_FAINT, zerolinewidth: 1.5 }),
  }));
}

/* -------------------------------------------- page 9 · model comparison - */
function comparisonCurves(models, order) {
  const roc = order.map((key, i) => ({
    type: 'scatter', mode: 'lines', name: models[key].name,
    x: models[key].curves.roc.fpr, y: models[key].curves.roc.tpr,
    line: { color: CAT[i % CAT.length], width: 2.2 },
    hovertemplate: `${models[key].name}<br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>`,
  }));
  roc.unshift({
    type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], showlegend: false,
    line: { color: LINE, width: 1, dash: 'dash' }, hoverinfo: 'skip',
  });
  draw('chart-cmp-roc', roc, base({
    margin: { l: 42, r: 14, t: 26, b: 34 },
    showlegend: true, legend: { orientation: 'h', y: 1.2, x: 0, font: { size: 9 } },
    xaxis: axis({ title: { text: 'false positive rate' } }),
    yaxis: axis({ title: { text: 'true positive rate' } }),
  }));

  const pr = order.map((key, i) => ({
    type: 'scatter', mode: 'lines', name: models[key].name,
    x: models[key].curves.pr.recall, y: models[key].curves.pr.precision,
    line: { color: CAT[i % CAT.length], width: 2.2 },
    hovertemplate: `${models[key].name}<br>recall %{x:.3f} · precision %{y:.3f}<extra></extra>`,
  }));
  const baseRate = models[order[0]].curves.positive_rate;
  pr.unshift({
    type: 'scatter', mode: 'lines', x: [0, 1], y: [baseRate, baseRate], showlegend: false,
    line: { color: LINE, width: 1, dash: 'dash' }, hoverinfo: 'skip',
  });
  draw('chart-cmp-pr', pr, base({
    margin: { l: 42, r: 14, t: 26, b: 34 },
    showlegend: true, legend: { orientation: 'h', y: 1.2, x: 0, font: { size: 9 } },
    xaxis: axis({ title: { text: 'recall' } }),
    yaxis: axis({ title: { text: 'precision' }, range: [0, 1] }),
  }));
}

window.Charts = {
  CAT, ACCENT, rupee, pct, resizeVisible,
  targetDonut, monthlyLine,
  edaTarget, edaCategory, edaAmount, edaNumeric,
  clusterScatter, elbowChart,
  probDistribution, rocChart, prChart, tradeoffChart,
  actualVsPredicted, residualPlot, comparisonCurves,
};
