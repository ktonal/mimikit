from ipywidgets import widgets as W

from ..features.functionals import MagSpec, MelSpec, MFCC, Chroma, \
    HarmonicSource, PercussiveSource, AutoConvolve, F0Filter,\
    NearestNeighborFilter, PCA, NMF, FactorAnalysis
from .. import ui as UI

__all__ = [
    "magspec_view",
    "melspec_view",
    "mfcc_view",
    "chroma_view",
    "harmonic_source_view",
    "percussive_source_view",
    "autoconvolve_view",
    "f0_filter_view",
    "nearest_neighbor_filter_view",
    "pca_view",
    "nmf_view",
    "factor_analysis_view"
]


def magspec_view(cfg: MagSpec):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_fft",
                 widget=UI.Labeled(
                     W.Label(value="N FFT: ", layout=label_layout),
                     W.IntText(value=cfg.n_fft, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("hop_length",
                 widget=UI.Labeled(
                     W.Label(value="hop length: ", layout=label_layout),
                     W.IntText(value=cfg.hop_length, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("center",
                 widget=UI.yesno_widget(
                     W.Label(value="center: ", layout=label_layout),
                     container=W.HBox(layout=param_layout),
                     initial_value=cfg.center,
                     buttons_layout=W.Layout(width="50%", margin="4px")
                 ), ),
        UI.Param("window",
                 widget=UI.EnumWidget(
                     W.Label(value="window: ", layout=label_layout),
                     [W.ToggleButton(description="None",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="hann",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="hamming",
                                     layout=W.Layout(width="33%", margin="4px")
                                     )
                      ],
                     W.HBox(layout=param_layout),
                     selected_index=0 if cfg.window is None else 1
                 ),
                 compute=lambda c, v: v if v != "None" else None
                 )
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Magnitude Spectrogram")
    return view


def melspec_view(cfg: MelSpec):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_mels",
                 widget=UI.Labeled(
                     W.Label(value="N Mels: ", layout=label_layout),
                     W.IntText(value=cfg.n_mels, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "MelSpectrogram")
    return view


def mfcc_view(cfg: MFCC):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_mfcc",
                 widget=UI.Labeled(
                     W.Label(value="N MFCC: ", layout=label_layout),
                     W.IntText(value=cfg.n_mfcc, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("dct_type",
                 widget=UI.EnumWidget(
                     W.Label(value="DCT Type: ", layout=label_layout),
                     [W.ToggleButton(description="1",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="2",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="3",
                                     layout=W.Layout(width="33%", margin="4px")
                                     )
                      ],
                     W.HBox(layout=param_layout),
                     selected_index=cfg.dct_type - 1
                 ),
                 compute=lambda c, v: int(v)
                 )
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "MFCC")
    return view


def chroma_view(cfg: Chroma):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_chroma",
                 widget=UI.Labeled(
                     W.Label(value="N Chroma: ", layout=label_layout),
                     W.IntText(value=cfg.n_chroma, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Chroma")
    return view


def harmonic_source_view(cfg: HarmonicSource):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("kernel_size",
                 widget=UI.Labeled(
                     W.Label(value="Kernel Size: ", layout=label_layout),
                     W.IntText(value=cfg.kernel_size, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("power",
                 widget=UI.Labeled(
                     W.Label(value="Power: ", layout=label_layout),
                     W.FloatText(value=cfg.power, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("margin",
                 widget=UI.Labeled(
                     W.Label(value="Margin: ", layout=label_layout),
                     W.FloatText(value=cfg.margin, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Harmonic Source")
    return view


def percussive_source_view(cfg: PercussiveSource):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("kernel_size",
                 widget=UI.Labeled(
                     W.Label(value="Kernel Size: ", layout=label_layout),
                     W.IntText(value=cfg.kernel_size, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("power",
                 widget=UI.Labeled(
                     W.Label(value="Power: ", layout=label_layout),
                     W.FloatText(value=cfg.power, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("margin",
                 widget=UI.Labeled(
                     W.Label(value="Margin: ", layout=label_layout),
                     W.FloatText(value=cfg.margin, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Percussive Source")
    return view


def autoconvolve_view(cfg: AutoConvolve):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("window_size",
                 widget=UI.Labeled(
                     W.Label(value="Window Size: ", layout=label_layout),
                     W.IntText(value=cfg.window_size, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "AutoConvolve")
    return view


def f0_filter_view(cfg: F0Filter):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_overtone",
                 widget=UI.Labeled(
                     W.Label(value="N Overtone: ", layout=label_layout),
                     W.IntText(value=cfg.n_overtone, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("n_undertone",
                 widget=UI.Labeled(
                     W.Label(value="N Undertone: ", layout=label_layout),
                     W.IntText(value=cfg.n_undertone, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("soft",
                 widget=UI.yesno_widget(
                     W.Label(value="Soft Filter: ", layout=label_layout),
                     container=W.HBox(layout=param_layout),
                     initial_value=cfg.soft,
                     buttons_layout=W.Layout(width="50%", margin="4px")
                 ), ),
        UI.Param("normalize",
                 widget=UI.yesno_widget(
                     W.Label(value="Normalize: ", layout=label_layout),
                     container=W.HBox(layout=param_layout),
                     initial_value=cfg.normalize,
                     buttons_layout=W.Layout(width="50%", margin="4px")
                 ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "F0 Filter")
    return view


def nearest_neighbor_filter_view(cfg: NearestNeighborFilter):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_neighbors",
                 widget=UI.Labeled(
                     W.Label(value="N Neighbors: ", layout=label_layout),
                     W.IntText(value=cfg.n_neighbors, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("metric",
                 widget=UI.EnumWidget(
                     W.Label(value="Metric: ", layout=label_layout),
                     [W.ToggleButton(description="cosine",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="euclidean",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="manhattan",
                                     layout=W.Layout(width="33%", margin="4px")
                                     )
                      ],
                     W.HBox(layout=param_layout),
                     selected_index=["cosine", "euclidean", "manhattan"].index(cfg.metric)
                 ),
                 ),
        UI.Param("aggregate",
                 widget=UI.EnumWidget(
                     W.Label(value="Aggregate: ", layout=label_layout),
                     [W.ToggleButton(description="mean",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="median",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="max",
                                     layout=W.Layout(width="33%", margin="4px")
                                     )
                      ],
                     W.HBox(layout=param_layout),
                     selected_index=["mean", "median", "max"].index(cfg.aggregate)
                 ),
                 ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Nearest Neighbor Filter")
    return view


def pca_view(cfg: PCA):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_components",
                 widget=UI.Labeled(
                     W.Label(value="N Components: ", layout=label_layout),
                     W.IntText(value=cfg.n_components, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled(
                     W.Label(value="Random Seed: ", layout=label_layout),
                     W.IntText(value=cfg.random_seed, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "PCA")
    return view


def nmf_view(cfg: NMF):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_components",
                 widget=UI.Labeled(
                     W.Label(value="N Components: ", layout=label_layout),
                     W.IntText(value=cfg.n_components, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("tol",
                 widget=UI.Labeled(
                     W.Label(value="Tolerance: ", layout=label_layout),
                     W.FloatText(value=cfg.tol, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("max_iter",
                 widget=UI.Labeled(
                     W.Label(value="Max Iter: ", layout=label_layout),
                     W.IntText(value=cfg.max_iter, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled(
                     W.Label(value="Random Seed: ", layout=label_layout),
                     W.IntText(value=cfg.random_seed, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "NMF")
    return view


def factor_analysis_view(cfg: FactorAnalysis):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_components",
                 widget=UI.Labeled(
                     W.Label(value="N Components: ", layout=label_layout),
                     W.IntText(value=cfg.n_components, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("tol",
                 widget=UI.Labeled(
                     W.Label(value="Tolerance: ", layout=label_layout),
                     W.FloatText(value=cfg.tol, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("max_iter",
                 widget=UI.Labeled(
                     W.Label(value="Max Iter: ", layout=label_layout),
                     W.IntText(value=cfg.max_iter, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled(
                     W.Label(value="Random Seed: ", layout=label_layout),
                     W.IntText(value=cfg.random_seed, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Factor Analysis")
    return view
