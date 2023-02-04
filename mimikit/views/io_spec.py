from ipywidgets import widgets as W
from .. import ui as UI

from ..io_spec import IOSpec


__all__ = [
  "mulaw_io_view",
  "yt_io_view",
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


def yt_io_view(cfg: IOSpec.YtIOConfig):
    view = UI.ConfigView(
        cfg,
        UI.Param(
            name='sr',
            widget=UI.Labeled("Sample Rate",
                              W.IntText(value=cfg.sr))
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
            name='max_scale',
            widget=UI.Labeled(
                "Maximum std",
                W.FloatSlider(value=cfg.max_scale, min=1e-4, max=1., step=0.0001)
            )
        ),
        UI.Param(
            name='beta',
            widget=UI.Labeled(
                "Sigmoid dilation factor",
                W.FloatSlider(value=cfg.beta, min=1e-4, max=1., step=0.0001)
            )
        ),
        UI.Param(
            name='weight_variance',
            widget=UI.Labeled(
                "Variance penalty",
                W.FloatSlider(value=cfg.weight_variance, min=1e-4, max=1., step=0.0001)
            )
        ),
        UI.Param(
            name='weight_l1',
            widget=UI.Labeled(
                "Reconstruction penalty",
                W.FloatSlider(value=cfg.weight_l1, min=1e-4, max=1., step=0.0001)
            )
        ),
        UI.Param(
            name='n_components',
            widget=UI.Labeled(
                "N random components",
                UI.pw2_widget(cfg.n_components)
            )
        ),
        UI.Param(
            name='objective_type',
            widget=UI.EnumWidget(
                "Output distribution",
                ["logistic", "laplace", "gaussian"],
                selected_index=["logistic_dist", "laplace_dist", "gaussian_dist"].index(cfg.objective_type)
            ),
            setter=lambda c, ev: ev + "_dist"
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
