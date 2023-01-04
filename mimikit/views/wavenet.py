import ipywidgets as W
from .. import ui as UI
from ..networks.wavenet_v2 import WaveNet

__all__ = [
    "wavenet_view"
]


def wavenet_view(cfg: WaveNet.Config):
    label_layout = W.Layout(min_width="max-content", margin="0 0 0 auto")
    param_layout = W.Layout(width="100%", margin="8px 0 8px 0")
    view = UI.ConfigView(
        cfg,
        UI.Param(name='kernel_sizes',
                 widget=UI.Labeled(
                     W.Label("kernel size",
                             layout=label_layout),
                     W.IntText(value=cfg.kernel_sizes[0],
                               layout={"width": "100px", }),
                     W.HBox(layout=param_layout)
                 ),
                 compute=lambda conf, ev: (ev,)),
        UI.Param(name="blocks",
                 widget=UI.Labeled(
                     W.Label(value="layers per block",
                             layout=label_layout),
                     W.Text(value=str(cfg.blocks)[1:-1], layout=W.Layout(width="75%", )),
                     W.HBox(layout=param_layout), ),
                 compute=lambda c, v: tuple(map(int, (s for s in v.split(",") if s not in ("", " "))))
                 ),
        UI.Param(name="dims_dilated",
                 widget=UI.pw2_widget(
                     W.Label(value="$N$ units per layer: ",
                             layout=label_layout, ),
                     W.Text(value=str(cfg.dims_dilated[0]),
                            layout=W.Layout(width="25%"), disabled=False),
                     W.Button(icon="plus", layout=W.Layout(width="25%")),
                     W.Button(icon="minus", layout=W.Layout(width="25%")),
                     W.HBox(layout=param_layout)),
                 compute=lambda c, v: (int(v),)
                 ),
        UI.Param(name='groups',
                 widget=UI.pw2_widget(
                     W.Label(value="groups of units",
                             layout=label_layout, ),
                     W.Text(value=str(cfg.groups),
                            layout=W.Layout(width="25%"), disabled=False),
                     W.Button(icon="plus", layout=W.Layout(width="25%")),
                     W.Button(icon="minus", layout=W.Layout(width="25%")),
                     W.HBox(layout=param_layout))
                 ),
        UI.Param(name="residuals_dim",
                 widget=UI.yesno_widget(
                     label=W.Label(value="use residuals", layout=label_layout),
                     container=W.HBox(layout=param_layout),
                     initial_value=cfg.residuals_dim is not None,
                     buttons_layout=W.Layout(width="27.5%")),
                 compute=lambda conf, ev: conf.dims_dilated[0] if ev else None
                 ),
        UI.Param(name="skips_dim",
                 widget=UI.yesno_widget(
                     label=W.Label(value="use skips", layout=label_layout),
                     container=W.HBox(layout=param_layout),
                     initial_value=cfg.residuals_dim is not None,
                     buttons_layout=W.Layout(width="27.5%")),
                 compute=lambda conf, ev: conf.dims_dilated[0] if ev else None
                 ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                selected_index=0, layout=W.Layout(margin="0 auto 0 0", width="100%"))
    view.set_title(0, "WaveNet Config")
    return view
