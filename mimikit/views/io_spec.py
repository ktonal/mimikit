from ipywidgets import widgets as W
from .. import ui as UI

from ..io_spec import IOSpec


__all__ = [
  "mulaw_io_view",
  "magspec_io_view"
]


def mulaw_io_view(cfg: IOSpec.MuLawIOConfig):
    view = UI.ConfigView(
        cfg,
        UI.Param(
            name='sr',
            widget=UI.Labeled("Sample Rate",
                              W.IntText(value=cfg.sr))
        ),
        UI.Param(
            name='q_levels',
            widget=UI.Labeled(
                "Quantization Levels",
                UI.pw2_widget(cfg.q_levels)
            )
        ),
        UI.Param(
            name="compression",
            widget=UI.Labeled(
                "Compression",
                W.FloatSlider(value=cfg.compression, min=0.001, max=2., step=0.01)
            )
        ),
        UI.Param(
            name='mlp_dim',
            widget=UI.Labeled("Final Layer Dim", UI.pw2_widget(cfg.mlp_dim))
        ),
        UI.Param(
            name='n_mlp_layer',
            widget=UI.Labeled(
                "N hidden final layers",
                W.IntText(cfg.n_mlp_layers)
            )
        ),
        UI.Param(
            name='min_temperature',
            widget=UI.Labeled(
                "Minimum temperature",
                W.FloatSlider(value=cfg.min_temperature, min=1e-4, max=1., step=0.0001)
            )
        )
    ).as_widget(lambda children, **kwargs: W.VBox(children=children))
    return view


def magspec_io_view(cfg: IOSpec.MagSpecIOConfig):
    view = UI.ConfigView(
        cfg,
        UI.Param(
            name='sr',
            widget=UI.Labeled("Sample Rate",
                              W.IntText(value=cfg.sr))
        ),
        UI.Param("n_fft",
                 widget=UI.Labeled("N FFT: ",
                                   W.IntText(value=cfg.n_fft),
                                   ), ),
        UI.Param("hop_length",
                 widget=UI.Labeled("hop length: ",
                                   W.IntText(value=cfg.hop_length),
                                   ), ),
        UI.Param(
            name='activation',
            widget=UI.EnumWidget(
                "Output Activation",
                ["Abs", "ScaledSigmoid"],
                selected_index=["Abs", "ScaledSigmoid"].index(cfg.activation)
            ),
        )
    ).as_widget(lambda children, **kwargs: W.VBox(children=children))
    return view
