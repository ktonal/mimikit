from ipywidgets import widgets as W

from ..extract.clusters import QCluster, GCluster, KMeans, SpectralClustering
from .. import ui as UI

__all__ = [
    "qcluster_view",
    "gcluster_view",
    "argmax_view",
    "kmeans_view",
    "spectral_clustering_view"
]


def qcluster_view(cfg: QCluster):

    view = UI.ConfigView(
        cfg,
        UI.Param("metric",
                 widget=UI.EnumWidget("Metric: ",
                                      ["cosine",
                                       "euclidean",
                                       "manhattan", ],
                                      selected_index=["cosine", "euclidean", "manhattan"].index(cfg.metric)
                                      ),
                 ),
        UI.Param("n_neighbors",
                 widget=UI.Labeled("N Neighbors: ",
                                   W.IntText(value=cfg.n_neighbors, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("cores_prop",
                 widget=UI.Labeled("Proportion of Cores: ",
                                   W.FloatText(value=cfg.cores_prop, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("core_neighborhood_size",
                 widget=UI.Labeled("N Neighbors per Core: ",
                                   W.IntText(value=cfg.core_neighborhood_size,
                                             layout=dict(margin='4px', width='auto')),
                                   ), ),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0"), selected_index=0)

    view.set_title(0, "Quantile Clustering")
    return view


def gcluster_view(cfg: GCluster):

    view = UI.ConfigView(
        cfg,
        UI.Param("metric",
                 widget=UI.EnumWidget("Metric: ",
                                      ["cosine",
                                       "euclidean", ],
                                      selected_index=["cosine", "euclidean"].index(cfg.metric)
                                      ),
                 ),
        UI.Param("n_means",
                 widget=UI.Labeled("N Means: ",
                                   W.IntText(value=cfg.n_means, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("n_iter",
                 widget=UI.Labeled("N Iter: ",
                                   W.IntText(value=cfg.n_iter,
                                             layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param(name="max_lr",
                 widget=UI.Labeled("Learning Rate: ",
                                   W.FloatSlider(
                                       value=1e-3, min=1e-5, max=1e-2, step=.00001,
                                       readout_format=".2e",
                                       layout={"width": "75%"}
                                   ),
                                   )),
        UI.Param(name="betas",
                 widget=UI.Labeled("Beta 1",
                                   W.FloatLogSlider(
                                       value=.9, min=-.75, max=0., step=.001, base=2,
                                       layout={"width": "75%"}),
                                   ),
                 setter=lambda conf, ev: (ev, conf.betas[1])),
        UI.Param(name="betas",
                 widget=UI.Labeled("Beta 2",
                                   W.FloatLogSlider(
                                       value=.9, min=-.75, max=0., step=.001, base=2,
                                       layout={"width": "75%"}),
                                   ),
                 setter=lambda conf, ev: (conf.betas[0], ev)),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0"), selected_index=0)

    view.set_title(0, "Grid of Means Clustering")
    return view


def argmax_view(cfg=None):
    view = W.Accordion(children=[W.Label(value="no parameters to set", layout=dict(width='auto'))],
                       layout=W.Layout(margin="0"), selected_index=0)
    view.set_title(0, "Arg Max Clustering")
    return view


def kmeans_view(cfg: KMeans):

    view = UI.ConfigView(
        cfg,
        UI.Param("n_clusters",
                 widget=UI.Labeled("N Components: ",
                                   W.IntText(value=cfg.n_clusters, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("n_init",
                 widget=UI.Labeled("N Init: ",
                                   W.IntText(value=cfg.n_init, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("max_iter",
                 widget=UI.Labeled("Max Iter: ",
                                   W.IntText(value=cfg.max_iter, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled("Random Seed: ",
                                   W.IntText(value=cfg.random_seed, layout=dict(margin='4px', width='auto')),
                                   ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0"), selected_index=0)

    view.set_title(0, "KMeans Clustering")
    return view


def spectral_clustering_view(cfg: SpectralClustering):

    view = UI.ConfigView(
        cfg,
        UI.Param("n_clusters",
                 widget=UI.Labeled("N Clusters: ",
                                   W.IntText(value=cfg.n_clusters, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("n_init",
                 widget=UI.Labeled("N Init: ",
                                   W.IntText(value=cfg.n_init, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("n_neighbors",
                 widget=UI.Labeled("N Neighbors: ",
                                   W.IntText(value=cfg.n_neighbors, layout=dict(margin='4px', width='auto')),
                                   ), ),
        UI.Param("random_seed",
                 widget=UI.Labeled("Random Seed: ",
                                   W.IntText(value=cfg.random_seed, layout=dict(margin='4px', width='auto')),
                                   ), )

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0"), selected_index=0)

    view.set_title(0, "Spectral Clustering Clustering")
    return view
