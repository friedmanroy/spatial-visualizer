"""A collapsible sidebar section that opens and closes cheaply.

Panel's own `pn.Card` rebuilds nearly every view on the page in the browser each time it
opens or closes - the plots included - which froze the Spatial tab for ~0.5-1 s per click
on a large sample. Showing or hiding a Column (`visible`) only re-lays the page out, about 7x
faster. This looks like a card: a clickable header (▸ closed, ▾ open) over the content.
"""
from __future__ import annotations

import param
import panel as pn

_HEADER_CSS = """
:host .bk-btn {
    text-align: left;
    font-weight: 600;
    font-size: 14px;
    padding: 6px 10px;
    border: none;
    border-radius: 4px;
    background: #f5f5f5;
}
:host .bk-btn:hover { background: #ececec; }
"""


class Collapsible(pn.viewable.Viewer):
    """`objects` under a header titled `title`; `collapsed` opens/closes it (and can be watched)."""

    collapsed = param.Boolean(default=True)

    def __init__(self, *objects, title: str, collapsed: bool = True, margin=(4, 5), **params):
        super().__init__(collapsed=collapsed, **params)
        self.title = title
        self._header = pn.widgets.Button(
            name=self._label(), sizing_mode="stretch_width", margin=0, stylesheets=[_HEADER_CSS],
        )
        self._header.on_click(lambda event: setattr(self, "collapsed", not self.collapsed))
        self._body = pn.Column(*objects, visible=not collapsed, sizing_mode="stretch_width", margin=(4, 0, 2, 0))
        self._layout = pn.Column(
            self._header, self._body, sizing_mode="stretch_width", margin=margin,
            styles={"border": "1px solid #e0e0e0", "border-radius": "4px"},
        )

    def _label(self) -> str:
        return f"{'▸' if self.collapsed else '▾'}  {self.title}"

    @param.depends("collapsed", watch=True)
    def _on_collapsed(self):
        with pn.io.hold():
            self._body.visible = not self.collapsed
            self._header.name = self._label()

    def __panel__(self):
        return self._layout
