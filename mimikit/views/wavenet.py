import ipywidgets as W
from .. import ui as UI
from ..networks.wavenet_v2 import WaveNet

__all__ = [
    "wavenet_view"
]


def wavenet_view(cfg: WaveNet.Config):
    view = UI.ConfigView(
        cfg,
        UI.Param(name='kernel_sizes',
                 widget=UI.Labeled(
                     "kernel size",
                     W.IntText(value=cfg.kernel_sizes[0]),
                 ),
                 setter=lambda conf, ev: (ev,)),
        UI.Param(name="blocks",
                 widget=UI.Labeled(
                     "layers per block",
                     W.Text(value=str(cfg.blocks)[1:-1])
                 ),
                 setter=lambda c, v: tuple(map(int, (s for s in v.split(",") if s not in ("", " "))))
                 ),
        UI.Param(name="dims_dilated",
                 widget=UI.Labeled(
                     "$N$ units per layer: ",
                     UI.pw2_widget(cfg.dims_dilated[0]),
                 ),
                 setter=lambda c, v: (int(v),)
                 ),
        UI.Param(name="dims_1x1",
                 widget=UI.Labeled(
                     "$N$ units per conditioning layer: ",
                     UI.pw2_widget(cfg.dims_dilated[0]),
                 ),
                 setter=lambda c, v: (int(v),)
                 ),
        UI.Param(name="residual_dim",
                 widget=UI.Labeled(
                     "$N$ units per residual layer: ",
                     UI.pw2_widget(cfg.dims_dilated[0]),
                 ),
                 setter=lambda c, v: (int(v),)
                 ),
        UI.Param(name="apply_residuals",
                 widget=UI.Labeled(
                     "use residuals",
                     UI.yesno_widget(initial_value=cfg.residuals_dim is not None),
                 ),
                 setter=lambda conf, ev: conf.dims_dilated[0] if ev else None
                 ),
        UI.Param(name="skips_dim",
                 widget=UI.Labeled(
                     "$N$ units per skip layer: ",
                     UI.pw2_widget(cfg.dims_dilated[0]),
                 ),
                 setter=lambda c, v: (int(v),)
                 ),
        UI.Param(name='groups',
                 widget=UI.Labeled(
                     "groups of units",
                     UI.pw2_widget(cfg.groups),
                 )),

        UI.Param(name="pad_side",
                 widget=UI.Labeled(
                     "use padding",
                     UI.yesno_widget(initial_value=bool(cfg.pad_side)),
                 ),
                 setter=lambda conf, ev: int(ev)
                 ),
        UI.Param(name="bias",
                 widget=UI.Labeled(
                     "use bias",
                     UI.yesno_widget(initial_value=cfg.bias),
                 ),
                 ),
        UI.Param(name="use_fast_generate",
                 widget=UI.Labeled(
                     "use fast generate",
                     UI.yesno_widget(initial_value=cfg.use_fast_generate),
                 ),
                 ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                selected_index=0, layout=W.Layout(margin="0 auto 0 0", width="100%"))
    view.set_title(0, "WaveNet Config")
    return view
