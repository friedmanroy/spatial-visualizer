"""Hover tooltips for the cell plots: what the cell (or hex bin) under the mouse is -
its category in the current field (and, for suggested cell types, the confidence in
it), or its values for the genes / pathways being shown.

Only numbers go into the plot's data source (category codes, confidences, channel
values), which Bokeh sends to the browser as compact binary arrays; one small JS
formatter turns them into text using labels sent once per field change.
"""
from __future__ import annotations

import numpy as np
from bokeh.models import CustomJSHover, HoverTool

#: hover columns in the plot's ColumnDataSource, all one value per drawn cell / bin
COLUMNS = ("hv_code", "hv_conf", "hv_share", "hv_n", "hv_r", "hv_g", "hv_b")

_JS = """
const i = special_vars.index;
const d = source.data;
const m = meta;
const lines = [];
const pct = (v) => Math.round(v * 100) + '%';
if (m.binned) lines.push(d.hv_n[i] + ' cells in this bin');
if (m.kind === 'field') {
    const c = d.hv_code[i];
    let text = m.field + ': ' + (c >= 0 ? m.labels[c] : (m.binned ? 'none selected' : 'missing'));
    if (m.binned && c >= 0) text += '  (' + pct(d.hv_share[i]) + ' of its cells)';
    lines.push(text);
    const conf = d.hv_conf[i];
    if (m.has_conf && !isNaN(conf)) lines.push('Confidence in this cell type: ' + pct(conf) + (m.binned ? ' (mean)' : ''));
} else if (m.kind === 'channels') {
    const cols = {R: d.hv_r, G: d.hv_g, B: d.hv_b};
    for (const [ch, name] of m.channels) {
        const v = cols[ch][i];
        lines.push(name + ': ' + (isNaN(v) ? 'not shown' : v.toFixed(2)) + (m.binned ? ' (mean)' : ''));
    }
}
return lines.join('\\n');
"""


def empty_columns(n: int) -> dict[str, np.ndarray]:
    nan = np.full(n, np.nan, dtype=np.float32)
    return {"hv_code": np.full(n, -1, dtype=np.int32), "hv_conf": nan, "hv_share": nan,
            "hv_n": np.ones(n, dtype=np.int32), "hv_r": nan, "hv_g": nan, "hv_b": nan}


def add_hover(fig, renderers, source) -> CustomJSHover:
    """Attaches the tooltip to `renderers`; returns the formatter whose `args['meta']`
    describes what's currently shown (see `ColorControls.hover_data`)."""
    formatter = CustomJSHover(args={"source": source, "meta": {"kind": "none"}}, code=_JS)
    fig.add_tools(HoverTool(
        renderers=list(renderers),
        tooltips='<div style="white-space:pre-line;font-size:12px;max-width:320px">@x{info}</div>',
        formatters={"@x": formatter},
    ))
    return formatter


def set_meta(formatter: CustomJSHover, meta: dict):
    if formatter.args.get("meta") != meta:
        formatter.args = {**formatter.args, "meta": meta}
