from ipywidgets import widgets as W

from ..features.functionals import MagSpec, MelSpec, MFCC, Chroma, \
    HarmonicSource, PercussiveSource, AutoConvolve, F0Filter, \
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

    view = UI.ConfigView(
        cfg,
        UI.Param("n_fft",
                 widget=UI.Labeled("N FFT: ",
                                   W.IntText(value=cfg.n_fft, layout=dict(width='auto')),
                                   ), ),
        UI.Param("hop_length",
                 widget=UI.Labeled("hop length: ",
                                   W.IntText(value=cfg.hop_length, layout=dict(width='auto')),
                                   ), ),
        UI.Param("center",
                 widget=
                 UI.Labeled("center: ", UI.yesno_widget(initial_value=cfg.center),
                            ), ),
        UI.Param("window",
                 widget=UI.EnumWidget("window: ",
                                      ["None", "hann", "hamming", ],
                                      selected_index=0 if cfg.window is None else 1
                                      ),
                 setter=lambda c, v: v if v != "None" else None
                 )
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 0 0 0"), selected_index=0)

    view.set_title(0, "Magnitude Spectrogram")
    return view


def melspec_view(cfg: MelSpec):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_mels",
                 widget=UI.Labeled("N Mels: ",
                                   W.IntText(value=cfg.n_mels, layout=dict(width='auto')),
                                   ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "MelSpectrogram")
    return view


def mfcc_view(cfg: MFCC):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_mfcc",
                 widget=UI.Labeled("N MFCC: ",
                                   W.IntText(value=cfg.n_mfcc, layout=dict(width='auto')),
                                   ), ),
        UI.Param("dct_type",
                 widget=UI.EnumWidget("DCT Type: ",
                                      ["1", "2", "3", ],
                                      selected_index=cfg.dct_type - 1
                                      ),
                 setter=lambda c, v: int(v)
                 )
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "MFCC")
    return view


def chroma_view(cfg: Chroma):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_chroma",
                 widget=UI.Labeled("N Chroma: ",
                                   W.IntText(value=cfg.n_chroma, layout=dict(width='auto')),
                                   ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "Chroma")
    return view


def harmonic_source_view(cfg: HarmonicSource):
    view = UI.ConfigView(
        cfg,
        UI.Param("kernel_size",
                 widget=UI.Labeled("Kernel Size: ",
                                   W.IntText(value=cfg.kernel_size, layout=dict(width='auto')),
                                   ), ),
        UI.Param("power",
                 widget=UI.Labeled("Power: ",
                                   W.FloatText(value=cfg.power, layout=dict(width='auto')),
                                   ), ),
        UI.Param("margin",
                 widget=UI.Labeled("Margin: ",
                                   W.FloatText(value=cfg.margin, layout=dict(width='auto')),
                                   ), ),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "Harmonic Source")
    return view


def percussive_source_view(cfg: PercussiveSource):
    view = UI.ConfigView(
        cfg,
        UI.Param("kernel_size",
                 widget=UI.Labeled("Kernel Size: ",
                                   W.IntText(value=cfg.kernel_size, layout=dict(width='auto')),
                                   ), ),
        UI.Param("power",
                 widget=UI.Labeled("Power: ",
                                   W.FloatText(value=cfg.power, layout=dict(width='auto')),
                                   ), ),
        UI.Param("margin",
                 widget=UI.Labeled("Margin: ",
                                   W.FloatText(value=cfg.margin, layout=dict(width='auto')),
                                   ), ),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "Percussive Source")
    return view


def autoconvolve_view(cfg: AutoConvolve):
    view = UI.ConfigView(
        cfg,
        UI.Param("window_size",
                 widget=UI.Labeled("Window Size: ",
                                   W.IntText(value=cfg.window_size, layout=dict(width='auto')),
                                   ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "AutoConvolve")
    return view


def f0_filter_view(cfg: F0Filter):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_overtone",
                 widget=UI.Labeled("N Overtone: ",
                                   W.IntText(value=cfg.n_overtone, layout=dict(width='auto')),
                                   ), ),
        UI.Param("n_undertone",
                 widget=UI.Labeled("N Undertone: ",
                                   W.IntText(value=cfg.n_undertone, layout=dict(width='auto')),
                                   ), ),
        UI.Param("soft",
                 widget=UI.Labeled("Soft Filter: ",
                                   UI.yesno_widget(initial_value=cfg.soft, )), ),
        UI.Param("normalize",
                 widget=UI.Labeled("Normalize: ",
                                   UI.yesno_widget(initial_value=cfg.normalize)
                                   ), ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "F0 Filter")
    return view


def nearest_neighbor_filter_view(cfg: NearestNeighborFilter):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_neighbors",
                 widget=UI.Labeled("N Neighbors: ",
                                   W.IntText(value=cfg.n_neighbors, layout=dict(width='auto')),
                                   ), ),
        UI.Param("metric",
                 widget=UI.EnumWidget("Metric: ",
                                      ["cosine",
                                       "euclidean",
                                       "manhattan",
                                       ],
                                      selected_index=["cosine", "euclidean", "manhattan"].index(cfg.metric)
                                      ),
                 ),
        UI.Param("aggregate",
                 widget=UI.EnumWidget("Aggregate: ",
                                      ["mean",
                                       "median",
                                       "max",
                                       ],
                                      selected_index=["mean", "median", "max"].index(cfg.aggregate)
                                      ),
                 ),
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "Nearest Neighbor Filter")
    return view


def pca_view(cfg: PCA):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_components",
                 widget=UI.Labeled("N Components: ",
                                   W.IntText(value=cfg.n_components, layout=dict(width='auto')),
                                   ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled("Random Seed: ",
                                   W.IntText(value=cfg.random_seed, layout=dict(width='auto')),
                                   ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "PCA")
    return view


def nmf_view(cfg: NMF):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_components",
                 widget=UI.Labeled("N Components: ",
                                   W.IntText(value=cfg.n_components, layout=dict(width='auto')),
                                   ), ),
        UI.Param("tol",
                 widget=UI.Labeled("Tolerance: ",
                                   W.FloatText(value=cfg.tol, layout=dict(width='auto')),
                                   ), ),
        UI.Param("max_iter",
                 widget=UI.Labeled("Max Iter: ",
                                   W.IntText(value=cfg.max_iter, layout=dict(width='auto')),
                                   ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled("Random Seed: ",
                                   W.IntText(value=cfg.random_seed, layout=dict(width='auto')),
                                   ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "NMF")
    return view


def factor_analysis_view(cfg: FactorAnalysis):
    view = UI.ConfigView(
        cfg,
        UI.Param("n_components",
                 widget=UI.Labeled("N Components: ",
                                   W.IntText(value=cfg.n_components, layout=dict(width='auto')),
                                   ), ),
        UI.Param("tol",
                 widget=UI.Labeled("Tolerance: ",
                                   W.FloatText(value=cfg.tol, layout=dict(width='auto')),
                                   ), ),
        UI.Param("max_iter",
                 widget=UI.Labeled("Max Iter: ",
                                   W.IntText(value=cfg.max_iter, layout=dict(width='auto')),
                                   ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled("Random Seed: ",
                                   W.IntText(value=cfg.random_seed, layout=dict(width='auto')),
                                   ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0"), selected_index=0)

    view.set_title(0, "Factor Analysis")
    return view
