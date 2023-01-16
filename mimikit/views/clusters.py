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
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
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
        UI.Param("n_neighbors",
                 widget=UI.Labeled(
                     W.Label(value="N Neighbors: ", layout=label_layout),
                     W.IntText(value=cfg.n_neighbors, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("cores_prop",
                 widget=UI.Labeled(
                     W.Label(value="Proportion of Cores: ", layout=label_layout),
                     W.FloatText(value=cfg.cores_prop, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("core_neighborhood_size",
                 widget=UI.Labeled(
                     W.Label(value="N Neighbors per Core: ", layout=label_layout),
                     W.IntText(value=cfg.core_neighborhood_size,
                               layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Quantile Clustering")
    return view


def gcluster_view(cfg: GCluster):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("metric",
                 widget=UI.EnumWidget(
                     W.Label(value="Metric: ", layout=label_layout),
                     [W.ToggleButton(description="cosine",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      W.ToggleButton(description="euclidean",
                                     layout=W.Layout(width="33%", margin="4px")
                                     ),
                      ],
                     W.HBox(layout=param_layout),
                     selected_index=["cosine", "euclidean"].index(cfg.metric)
                 ),
                 ),
        UI.Param("n_means",
                 widget=UI.Labeled(
                     W.Label(value="N Means: ", layout=label_layout),
                     W.IntText(value=cfg.n_means, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("n_iter",
                 widget=UI.Labeled(
                     W.Label(value="N Iter: ", layout=label_layout),
                     W.IntText(value=cfg.n_iter,
                               layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param(name="max_lr",
                 widget=UI.Labeled(
                     W.Label(value="Learning Rate: ",
                             layout=label_layout
                             ),
                     W.FloatSlider(
                         value=1e-3, min=1e-5, max=1e-2, step=.00001,
                         readout_format=".2e",
                         layout={"width": "75%"}
                     ),
                     W.HBox(
                         layout=param_layout
                     )
                 )),
        UI.Param(name="betas",
                 widget=UI.Labeled(
                     W.Label(value="Beta 1", layout=label_layout),
                     W.FloatLogSlider(
                         value=.9, min=-.75, max=0., step=.001, base=2,
                         layout={"width": "75%"}),
                     W.HBox(layout=param_layout)
                 ),
                 compute=lambda conf, ev: (ev, conf.betas[1])),
        UI.Param(name="betas",
                 widget=UI.Labeled(
                     W.Label(value="Beta 2", layout=label_layout),
                     W.FloatLogSlider(
                         value=.9, min=-.75, max=0., step=.001, base=2,
                         layout={"width": "75%"}),
                     W.HBox(layout=param_layout)
                 ),
                 compute=lambda conf, ev: (conf.betas[0], ev)),

    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)

    view.set_title(0, "Grid of Means Clustering")
    return view


def argmax_view(cfg=None):
    view = W.Accordion(children=[W.Label(value="no parameters to set", layout=dict(width="100%"))],
                       layout=W.Layout(margin="0 auto 0 0", width="33%"), selected_index=0)
    view.set_title(0, "Arg Max Clustering")
    return view


def kmeans_view(cfg: KMeans):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_clusters",
                 widget=UI.Labeled(
                     W.Label(value="N Components: ", layout=label_layout),
                     W.IntText(value=cfg.n_clusters, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("n_init",
                 widget=UI.Labeled(
                     W.Label(value="N Init: ", layout=label_layout),
                     W.IntText(value=cfg.n_init, layout=dict(margin='4px', width="100%")),
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

    view.set_title(0, "KMeans Clustering")
    return view


def spectral_clustering_view(cfg: SpectralClustering):
    label_layout = W.Layout(min_width="max-content", margin="auto 12px auto auto")
    param_layout = W.Layout(width="100%", margin="6px 0 6px 0", display="flex")

    view = UI.ConfigView(
        cfg,
        UI.Param("n_clusters",
                 widget=UI.Labeled(
                     W.Label(value="N Components: ", layout=label_layout),
                     W.IntText(value=cfg.n_clusters, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("n_init",
                 widget=UI.Labeled(
                     W.Label(value="N Init: ", layout=label_layout),
                     W.IntText(value=cfg.n_init, layout=dict(margin='4px', width="100%")),
                     W.HBox([], layout=param_layout)
                 ), ),
        UI.Param("n_neighbors",
                 widget=UI.Labeled(
                     W.Label(value="Max Iter: ", layout=label_layout),
                     W.IntText(value=cfg.n_neighbors, layout=dict(margin='4px', width="100%")),
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

    view.set_title(0, "Spectral Clustering Clustering")
    return view
