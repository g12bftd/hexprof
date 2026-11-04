"""Self-contained HTML pages (no network needed) with interactive SVG charts.

Charts follow one method: the form is picked by the data's job (ranked bars for
magnitude, stacked bars for part-to-whole across graphs, a line for counters over
time, a one-hue heat strip for occupancy over time); categorical colours come from
a validated 8-slot palette in fixed order and follow the entity, never its rank;
every chart has a hover/focus tooltip and a table view; light and dark modes.
"""

from __future__ import annotations

import html
import json
from typing import Any

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --other: #b9b8b0;
  --q0: #eeede8; --q1: #cde2fb; --q2: #9ec5f4; --q3: #6da7ec; --q4: #3987e5; --q5: #256abf; --q6: #184f95; --q7: #0d366b;
  --warn: #fab219; --crit: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
    --other: #5b5a55;
    --q0: #252524; --q1: #104281; --q2: #184f95; --q3: #1c5cab; --q4: #256abf; --q5: #3987e5; --q6: #6da7ec; --q7: #b7d3f6;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  --other: #5b5a55;
  --q0: #252524; --q1: #104281; --q2: #184f95; --q3: #1c5cab; --q4: #256abf; --q5: #3987e5; --q6: #6da7ec; --q7: #b7d3f6;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 64px; }
header h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; overflow-wrap: anywhere; }
header p, .card .why, .note, .warnbox { overflow-wrap: anywhere; }
header p { margin: 0; color: var(--text-secondary); }
nav { margin-bottom: 12px; font-size: 13px; }
a { color: var(--s1); }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-top: 16px; }
.card h2 { font-size: 16px; font-weight: 600; margin: 0 0 2px; }
.card .why { color: var(--text-secondary); margin: 0 0 12px; font-size: 13px; max-width: 900px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile .sub { color: var(--muted); font-size: 12px; }
.chart { width: 100%; overflow-x: auto; }
.chart svg { display: block; }
svg text { fill: var(--text-secondary); font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
svg text.val { fill: var(--text-primary); font-variant-numeric: tabular-nums; }
svg text.axis { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .base { stroke: var(--axis); stroke-width: 1; }
svg .hit { fill: transparent; cursor: default; }
svg .hit:focus { outline: none; }
svg g.mark:hover rect.m, svg g.mark:focus rect.m { filter: brightness(1.12); }
svg g.mark:focus rect.m { stroke: var(--text-primary); stroke-width: 1; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 0 0 10px; font-size: 12px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.scale { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); margin-top: 8px; }
.scale .ramp { display: flex; height: 8px; width: 160px; border-radius: 4px; overflow: hidden; }
.scale .ramp b { flex: 1; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.tablewrap { max-height: 420px; overflow: auto; }
.badge { display: inline-block; padding: 0 6px; border-radius: 4px; font-size: 11px; border: 1px solid var(--border); color: var(--text-secondary); }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.warnbox { border-left: 3px solid var(--warn); padding: 6px 10px; margin-top: 12px; color: var(--text-secondary); font-size: 13px; }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.15);
  max-width: 360px; display: none; }
#tip .v { font-weight: 600; font-size: 14px; }
#tip .k { color: var(--text-secondary); }
#tip .row { display: flex; gap: 8px; align-items: center; }
#tip .row i { width: 12px; height: 2px; display: inline-block; }
</style>
</head>
<body>
<main id="root"></main>
<div id="tip" role="status" aria-live="polite"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const NS = "http://www.w3.org/2000/svg";
const SLOTS = ["var(--s1)","var(--s2)","var(--s3)","var(--s4)","var(--s5)","var(--s6)","var(--s7)","var(--s8)"];
const RAMP = ["var(--q0)","var(--q1)","var(--q2)","var(--q3)","var(--q4)","var(--q5)","var(--q6)","var(--q7)"];

function el(tag, attrs, parent) { const e = document.createElement(tag); for (const k in (attrs||{})) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function sv(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in (attrs||{})) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(tag, text, parent, cls) { const e = el(tag, cls ? {class: cls} : {}, parent); e.textContent = text; return e; }

const F = {
  num(v, d) { if (v == null || isNaN(v)) return "–"; if (v === 0) return "0"; const a = Math.abs(v);
    if (a >= 1e9) return (v/1e9).toFixed(d ?? 2) + "G"; if (a >= 1e6) return (v/1e6).toFixed(d ?? 2) + "M";
    if (a >= 1e4) return (v/1e3).toFixed(d ?? 1) + "K"; if (a >= 100) return v.toFixed(0); if (a >= 1) return v.toFixed(d ?? 2);
    return v.toPrecision(3); },
  bytes(v) { if (v == null || isNaN(v)) return "–"; const a = Math.abs(v);
    if (a >= 1073741824) return (v/1073741824).toFixed(2) + " GiB"; if (a >= 1048576) return (v/1048576).toFixed(1) + " MiB";
    if (a >= 1024) return (v/1024).toFixed(1) + " KiB"; return v.toFixed(0) + " B"; },
  pct(v) { return v == null || isNaN(v) ? "–" : v.toFixed(1) + "%"; },
  frac(v) { return v == null || isNaN(v) ? "–" : (100 * v).toFixed(1) + "%"; },
  us(v) { if (v == null || isNaN(v)) return "–"; if (Math.abs(v) >= 1e3) return (v/1e3).toFixed(2) + " ms"; return v.toFixed(1) + " µs"; },
  gbps(v) { return v == null || isNaN(v) ? "–" : v.toFixed(1) + " GB/s"; },
  str(v) { return v == null ? "–" : String(v); },
};
function fmtBy(kind) { return F[kind] || F.num; }

// ---- tooltip ----------------------------------------------------------------------------------------
const TIP = document.getElementById("tip");
function showTip(evt, value, label, extra) {
  TIP.replaceChildren();
  txt("div", value, TIP, "v"); txt("div", label, TIP, "k");
  for (const [k, v, color] of (extra || [])) {
    const r = el("div", {class: "row"}, TIP);
    if (color) { const i = el("i", {}, r); i.style.background = color; }
    txt("span", k + ": ", r, "k"); txt("span", v, r);
  }
  TIP.style.display = "block";
  let x = 0, y = 0;
  if (evt && evt.clientX != null && evt.type !== "focus") { x = evt.clientX; y = evt.clientY; }
  else if (evt && evt.target && evt.target.getBoundingClientRect) { const b = evt.target.getBoundingClientRect(); x = b.right; y = b.top; }
  const w = TIP.offsetWidth, h = TIP.offsetHeight;
  TIP.style.left = Math.min(x + 14, window.innerWidth - w - 8) + "px";
  TIP.style.top = Math.max(8, Math.min(y + 14, window.innerHeight - h - 8)) + "px";
}
function hideTip() { TIP.style.display = "none"; }
function hover(node, fn) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", e => fn(e)); node.addEventListener("focus", e => fn(e));
  node.addEventListener("pointerleave", hideTip); node.addEventListener("blur", hideTip);
}
function chartW(parent) { const cs = getComputedStyle(parent);
  const w = parent.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
  return Math.max(300, Math.floor(w || 900)); }
function edgeAnchor(x, W) { return x > W - 40 ? "end" : x < 40 ? "start" : "middle"; }
function truncate(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function niceTicks(max, n) {
  if (!(max > 0)) return [0];
  const raw = max / (n || 4), p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p, out = [];
  for (let v = 0; v <= max * 1.0001 + step * 0.01; v += step) out.push(v);
  return out;
}

// ---- table view (every chart has one) -----------------------------------------------------------------
function tableView(parent, cols, rows, open) {
  const det = el("details", open ? {open: ""} : {}, parent);
  txt("summary", "Table (" + rows.length + (rows.length === 1 ? " row)" : " rows)"), det);
  const wrap = el("div", {class: "tablewrap"}, det), t = el("table", {}, wrap), tr = el("tr", {}, el("thead", {}, t));
  for (const c of cols) txt("th", c.label, tr, c.kind && c.kind !== "str" ? "n" : "");
  const tb = el("tbody", {}, t);
  for (const r of rows) {
    const row = el("tr", {}, tb);
    for (const c of cols) {
      const v = r[c.key];
      if (c.link && r[c.link]) { const td = el("td", {}, row); const a = el("a", {href: r[c.link]}, td); a.textContent = F.str(v); }
      else txt("td", c.kind ? fmtBy(c.kind)(v) : F.str(v), row, c.kind && c.kind !== "str" ? "n" : "");
    }
  }
  return det;
}

// ---- ranked horizontal bars (one series, one hue) --------------------------------------------------------
function hbar(parent, rows, o) {
  o = o || {}; const f = fmtBy(o.kind), W = chartW(parent), labW = Math.min(o.labelWidth || 260, Math.round(W * 0.4));
  const rowH = 26, barH = 16, top = 6, plotW = W - labW - 90, H = top + rows.length * rowH + 22;
  const max = o.max != null ? o.max : Math.max(1e-12, ...rows.map(r => r.value || 0));
  const box = el("div", {class: "chart"}, parent), s = sv("svg", {width: W, height: H, role: "img", "aria-label": o.title || "bar chart"}, box);
  const ticks = niceTicks(max, 4), xMax = Math.max(max, ticks[ticks.length - 1]), x = v => labW + plotW * (v / xMax);
  for (const t of ticks) { sv("line", {x1: x(t), x2: x(t), y1: top, y2: H - 20, class: "grid"}, s);
    sv("text", {x: x(t), y: H - 6, "text-anchor": "middle", class: "axis"}, s).textContent = f(t); }
  sv("line", {x1: labW, x2: labW, y1: top, y2: H - 20, class: "base"}, s);
  rows.forEach((r, i) => {
    const y = top + i * rowH, g = sv("g", {class: "mark"}, s), w = Math.max(0, x(r.value || 0) - labW);
    sv("text", {x: labW - 8, y: y + rowH / 2 + 4, "text-anchor": "end"}, g).textContent = truncate(r.label, Math.floor(labW / 7));
    const color = r.color || o.color || SLOTS[0], rr = Math.min(4, w / 2);
    if (w > 0) sv("path", {class: "m", d: `M${labW},${y + (rowH - barH) / 2} h${w - rr} a${rr},${rr} 0 0 1 ${rr},${rr} v${barH - 2 * rr} a${rr},${rr} 0 0 1 -${rr},${rr} h-${w - rr} z`, fill: color}, g);
    sv("text", {x: labW + w + 6, y: y + rowH / 2 + 4, class: "val"}, g).textContent = f(r.value);
    const hit = sv("rect", {x: 0, y: y, width: W, height: rowH, class: "hit"}, g);
    hover(hit, e => showTip(e, f(r.value), r.label, r.extra || []));
  });
  return box;
}

// ---- stacked horizontal bars (several series, legend + 2px surface gaps) ----------------------------------
function stacked(parent, rows, series, o) {
  o = o || {}; const f = fmtBy(o.kind), W = chartW(parent), labW = Math.min(o.labelWidth || 220, Math.round(W * 0.4));
  const lg = el("div", {class: "legend"}, parent);
  series.forEach((sname, i) => { const sp = el("span", {}, lg); const sw = el("i", {}, sp); sw.style.background = sname === "Other" ? "var(--other)" : SLOTS[i % 8]; txt("span", sname, sp); });
  const rowH = 28, barH = 18, top = 4, plotW = W - labW - 90, H = top + rows.length * rowH + 22;
  const totals = rows.map(r => r.values.reduce((a, b) => a + (b || 0), 0));
  const max = o.max != null ? o.max : Math.max(1e-12, ...totals);
  const box = el("div", {class: "chart"}, parent), s = sv("svg", {width: W, height: H, role: "img", "aria-label": o.title || "stacked bar chart"}, box);
  const ticks = niceTicks(max, 4), xMax = Math.max(max, ticks[ticks.length - 1]), sc = v => plotW * (v / xMax);
  for (const t of ticks) { sv("line", {x1: labW + sc(t), x2: labW + sc(t), y1: top, y2: H - 20, class: "grid"}, s);
    sv("text", {x: labW + sc(t), y: H - 6, "text-anchor": "middle", class: "axis"}, s).textContent = f(t); }
  rows.forEach((r, i) => {
    const y = top + i * rowH + (rowH - barH) / 2; let xc = labW;
    sv("text", {x: labW - 8, y: y + barH / 2 + 4, "text-anchor": "end"}, s).textContent = truncate(r.label, Math.floor(labW / 7));
    r.values.forEach((v, j) => {
      if (!(v > 0)) return; const w = sc(v), g = sv("g", {class: "mark"}, s);
      const color = series[j] === "Other" ? "var(--other)" : SLOTS[j % 8];
      sv("rect", {class: "m", x: xc, y: y, width: Math.max(0.5, w - 2), height: barH, rx: 2, fill: color}, g);
      const hit = sv("rect", {x: xc, y: y - 4, width: Math.max(4, w), height: barH + 8, class: "hit"}, g);
      hover(hit, e => showTip(e, f(v) + (totals[i] ? "  (" + (100 * v / totals[i]).toFixed(1) + "%)" : ""), r.label + " · " + series[j]));
      xc += w;
    });
    sv("text", {x: xc + 6, y: y + barH / 2 + 4, class: "val"}, s).textContent = f(totals[i]);
  });
  return box;
}

// ---- line over time (counters): 2px line, 10% wash, crosshair snaps to the nearest sample --------------------
function line(parent, pts, o) {
  o = o || {}; const f = fmtBy(o.kind), W = chartW(parent), H = 220, L = 64, R = 16, T = 10, B = 26;
  if (!pts.length) { txt("p", "No samples.", parent, "note"); return; }
  const xf = !o.xunit || o.xunit === "µs" ? F.us : (v => F.num(v) + " " + o.xunit);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), x0 = xs[0], x1 = xs[xs.length - 1] || x0 + 1;
  const yMax = Math.max(1e-12, ...pts.map(p => p.length > 2 ? p[2] : p[1])), ticks = niceTicks(yMax, 4), yTop = ticks[ticks.length - 1];
  const X = v => L + (W - L - R) * ((v - x0) / Math.max(1e-12, x1 - x0)), Y = v => T + (H - T - B) * (1 - v / yTop);
  const box = el("div", {class: "chart"}, parent), s = sv("svg", {width: W, height: H, role: "img", "aria-label": o.title || "line chart"}, box);
  for (const t of ticks) { sv("line", {x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "grid"}, s);
    sv("text", {x: L - 6, y: Y(t) + 4, "text-anchor": "end", class: "axis"}, s).textContent = f(t); }
  for (const t of niceTicks(x1 - x0, 5)) { if (x0 + t > x1) break;
    sv("text", {x: X(x0 + t), y: H - 6, "text-anchor": edgeAnchor(X(x0 + t), W), class: "axis"}, s).textContent = xf(t); }
  let d = ""; pts.forEach((p, i) => { d += (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); });
  sv("path", {d: d + `L${X(x1)},${Y(0)}L${X(x0)},${Y(0)}Z`, fill: o.color || SLOTS[0], opacity: 0.1}, s);
  sv("path", {d: d, fill: "none", stroke: o.color || SLOTS[0], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, s);
  const cross = sv("line", {y1: T, y2: H - B, class: "base", visibility: "hidden"}, s);
  const dot = sv("circle", {r: 4, fill: o.color || SLOTS[0], stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden"}, s);
  const hit = sv("rect", {x: L, y: T, width: W - L - R, height: H - T - B, class: "hit"}, s);
  const at = e => { const r = s.getBoundingClientRect(), xv = x0 + (e.clientX - r.left - L) / (W - L - R) * (x1 - x0);
    let lo = 0, hi = xs.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < xv) lo = m; else hi = m; }
    return Math.abs(xs[lo] - xv) <= Math.abs(xs[hi] - xv) ? lo : hi; };
  hit.setAttribute("tabindex", "0");
  hit.addEventListener("pointermove", e => { const i = at(e), p = pts[i];
    cross.setAttribute("x1", X(p[0])); cross.setAttribute("x2", X(p[0])); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(p[0])); dot.setAttribute("cy", Y(p[1])); dot.setAttribute("visibility", "visible");
    showTip(e, f(p[1]), "t = " + xf(p[0] - x0), p.length > 2 ? [["max in bucket", f(p[2])]] : []); });
  hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); });
  return box;
}

// ---- occupancy heat strips: one row per lane, one-hue sequential ramp ------------------------------------------
function lanes(parent, tl, o) {
  o = o || {}; const W = chartW(parent), labW = Math.min(o.labelWidth || 220, Math.round(W * 0.35)), rowH = 18, gap = 4;
  const n = tl.lanes.length, bins = n ? tl.lanes[0].busy.length : 0, H = 6 + n * (rowH + gap) + 22, plotW = W - labW - 16;
  const box = el("div", {class: "chart"}, parent), s = sv("svg", {width: W, height: H, role: "img", "aria-label": o.title || "occupancy"}, box);
  const bw = plotW / Math.max(1, bins), span = tl.t1 - tl.t0, unitFmt = tl.unit === "µs" ? F.us : (v => F.num(v) + " " + tl.unit);
  tl.lanes.forEach((ln, i) => {
    const y = 6 + i * (rowH + gap);
    sv("text", {x: labW - 8, y: y + rowH / 2 + 4, "text-anchor": "end"}, s).textContent = truncate(ln.lane, Math.floor(labW / 7));
    ln.busy.forEach((b, j) => {
      const step = b <= 0 ? 0 : Math.min(7, 1 + Math.floor(b * 7));
      sv("rect", {x: labW + j * bw, y: y, width: Math.max(0.6, bw + 0.15), height: rowH, fill: RAMP[step]}, s);
    });
    const hit = sv("rect", {x: labW, y: y, width: plotW, height: rowH, class: "hit"}, s);
    hit.setAttribute("tabindex", "0");
    hit.addEventListener("pointermove", e => { const r = s.getBoundingClientRect(), j = Math.max(0, Math.min(bins - 1, Math.floor((e.clientX - r.left - labW) / bw)));
      showTip(e, F.frac(ln.busy[j]) + " busy", ln.lane, [["from", unitFmt(j * span / bins)], ["to", unitFmt((j + 1) * span / bins)]]); });
    hit.addEventListener("pointerleave", hideTip);
  });
  for (const t of niceTicks(span, 5)) { if (t > span) break;
    sv("text", {x: labW + plotW * t / span, y: H - 6, "text-anchor": edgeAnchor(labW + plotW * t / span, W), class: "axis"}, s).textContent = unitFmt(t); }
  const sc = el("div", {class: "scale"}, parent); txt("span", "idle", sc); const rp = el("span", {class: "ramp"}, sc);
  for (const c of RAMP) { const b = el("b", {}, rp); b.style.background = c; } txt("span", "fully busy", sc);
  return box;
}

// ---- page assembly --------------------------------------------------------------------------------------------
function card(root, title, why) { const c = el("section", {class: "card"}, root); txt("h2", title, c); if (why) txt("p", why, c, "why"); return c; }
function tiles(root, items) { const t = el("div", {class: "tiles"}, root);
  for (const it of items) { const d = el("div", {class: "tile"}, t); txt("div", it.label, d, "label"); txt("div", it.value, d, "value"); if (it.sub) txt("div", it.sub, d, "sub"); } }

function render(root, page) {
  if (page.nav) { const n = el("nav", {}, root); const a = el("a", {href: page.nav.href}, n); a.textContent = page.nav.text; }
  const h = el("header", {}, root); txt("h1", page.title, h); if (page.subtitle) txt("p", page.subtitle, h);
  for (const w of page.warnings || []) txt("div", w, root, "warnbox");
  if (page.tiles) tiles(root, page.tiles);
  for (const sec of page.sections) {
    const c = card(root, sec.title, sec.why);
    for (const b of sec.blocks) {
      if (b.type === "hbar") hbar(c, b.rows, b);
      else if (b.type === "stacked") stacked(c, b.rows, b.series, b);
      else if (b.type === "line") { if (b.subtitle) txt("p", b.subtitle, c, "note"); line(c, b.points, b); }
      else if (b.type === "lanes") lanes(c, b.timeline, b);
      else if (b.type === "note" && b.text) txt("p", b.text, c, "note");
      if (b.table) tableView(c, b.table.cols, b.table.rows, b.table.open);
    }
  }
  if (page.footer) txt("p", page.footer, root, "note");
}
render(document.getElementById("root"), D);
</script>
</body>
</html>
"""


def render_page(page: dict[str, Any]) -> str:
    data = json.dumps(page, default=_json_default, allow_nan=False).replace("</", "<\\/")
    return PAGE.replace("__TITLE__", html.escape(str(page.get("title", "Report")))).replace("__DATA__", data)


def _json_default(o: Any) -> Any:
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)
