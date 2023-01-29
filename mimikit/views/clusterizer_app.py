from typing import *
import dataclasses as dtc
from ipywidgets import widgets as W

from ..extract.clusters import *
from ..features.functionals import *
from .clusters import *
from .functionals import *


__all__ = [
    'ComposeTransformWidget',
    'ClusterWidget',
    'Segment'

]


@dtc.dataclass
class Segment:
    startTime: float
    endTime: float
    id: int
    color: str = '#ff640e'
    labelText: str = ""
    editable: bool = True
    duration: float = dtc.field(init=False)

    def __post_init__(self):
        self.duration = self.endTime - self.startTime

    def dict(self):
        return dtc.asdict(self)


@dtc.dataclass
class Meta:
    config_class: Type
    view_func: Callable
    requires: List[Type] = dtc.field(default_factory=lambda: [])
    only_once: bool = False

    def can_be_added(self, preceding_transforms: List[Type]):
        if not self.requires:
            return not preceding_transforms
        if self.requires[0] is Any and len(preceding_transforms) > 0:
            return True
        deps_fullfilled = self.requires == preceding_transforms
        if self.only_once:
            already_there = any(f is self.config_class for f in preceding_transforms)
        else:
            already_there = False
        return deps_fullfilled and not already_there


TRANSFORMS = {
    "magspec": Meta(MagSpec, magspec_view, [], True),
    "melspec": Meta(MelSpec, melspec_view, [MagSpec], True),
    "mfcc": Meta(MFCC, mfcc_view, [MagSpec, MelSpec], True),
    "chroma": Meta(Chroma, chroma_view, [MagSpec], True),
    "autoconvolve": Meta(AutoConvolve, autoconvolve_view, [Any], False),
    "f0 filter": Meta(F0Filter, f0_filter_view, [MagSpec], False),
    "nearest_neighbor_filter": Meta(NearestNeighborFilter, nearest_neighbor_filter_view, [Any]),
    "pca": Meta(PCA, pca_view, [Any]),
    "nmf": Meta(NMF, nmf_view, [Any]),
    "factor analysis": Meta(FactorAnalysis, factor_analysis_view, [Any])
}

CLUSTERINGS = {
    "grid of means": Meta(GCluster, gcluster_view, [], True),
    "quantile clustering": Meta(QCluster, qcluster_view, [], True),
    "argmax": Meta(ArgMax, argmax_view, [], True),
    "kmeans": Meta(KMeans, kmeans_view, [], True),
    "spectral clustering": Meta(SpectralClustering, spectral_clustering_view, [], True)
}


class ComposeTransformWidget:

    def __init__(self, width="500px"):
        self.transforms = []
        self.metas = []

        new_choice = W.Button(icon="fa-plus", layout=dict(margin="8px auto"))

        collapse_all = W.Button(description="collapse all",
                                layout=dict(width="max-content", margin="auto 4px auto 2px"))
        expand_all = W.Button(description="expand all", layout=dict(width="max-content", margin="auto auto auto 4px"))
        header = W.HBox(children=[
            W.HTML(value="<h4> Pre Processing Pipeline </h4>", layout=dict(margin="auto")),
            collapse_all, expand_all
        ])
        box = W.VBox(children=(header,
                               new_choice,),
                     layout=dict(width=width))

        def on_collapse(ev):
            for item in box.children:
                if isinstance(item, W.HBox) and isinstance(item.children[-1], W.Accordion):
                    item.children[-1].selected_index = None

        def on_expand(ev):
            for item in box.children:
                if isinstance(item, W.HBox) and isinstance(item.children[-1], W.Accordion):
                    item.children[-1].selected_index = 0

        collapse_all.on_click(on_collapse)
        expand_all.on_click(on_expand)

        choices = W.Select(options=self.get_possible_choices(), layout=dict(width="100%", margin="4px auto"))
        submit = W.Button(description="submit", layout=dict(width="max-content", margin="auto 8px"))
        cancel = W.Button(description="cancel", layout=dict(width="max-content", margin="auto 8px"))
        choice_box = W.VBox(children=(choices, W.HBox(children=(submit, cancel),
                                                      layout=dict(margin="4px auto"))),
                            layout=dict(width="calc(100% - 54px)", margin="auto 0 auto 54px"))

        def show_new_choice(ev):
            box.children = (*filter(lambda b: b is not new_choice, box.children), choice_box)
            new_choice.disabled = True

        def add_choice(ev):
            meta = TRANSFORMS[choices.value]
            cfg = meta.config_class()
            self.metas.append(meta)
            self.transforms.append(cfg)
            new_w = meta.view_func(cfg)
            new_w.layout = dict(width="95%", margin="auto")
            remove_w = W.Button(icon="fa-trash", layout=dict(width="50px", margin="auto 2px"))
            hbox = W.HBox(children=(remove_w, new_w), layout=dict(width="95%", margin="0 4px 4px 4px"))
            choices.options = self.get_possible_choices()

            def remove_cb(ev):
                keep = []
                for i, (t, m) in enumerate(zip(self.transforms, self.metas)):
                    is_el = t is cfg
                    requires_t = type(cfg) in m.requires or not meta.requires  # MagSpec kills everything!
                    if not is_el and not requires_t:
                        keep += [i]
                self.transforms = [self.transforms[i] for i in keep]
                box.children = (box.children[0],) + tuple(box.children[i + 1] for i in keep) + (box.children[-1],)
                choices.options = self.get_possible_choices()

            remove_w.on_click(remove_cb)
            box.children = (*filter(lambda b: b is not choice_box, box.children), hbox, new_choice)
            new_choice.disabled = False

        def cancel_new_choice(ev):
            box.children = (*filter(lambda b: b is not choice_box, box.children), new_choice)
            new_choice.disabled = False

        submit.on_click(add_choice)
        cancel.on_click(cancel_new_choice)
        new_choice.on_click(show_new_choice)
        self.widget = box

    def get_possible_choices(self):
        options = []
        ts = [*map(type, self.transforms)]
        for k, meta in TRANSFORMS.items():
            if meta.can_be_added(ts):
                options += [k]
        return options


class ClusterWidget:
    def __init__(self, width="400px"):
        self.cfg = None
        new_choice = W.Button(description="change algo",
                              layout=dict(width="max-content", margin="auto auto auto 12px"),
                              disabled=True)
        header = W.HBox(children=[
            W.HTML(value="<h4> Clustering Algo </h4>", layout=dict(margin="auto")),
            new_choice
        ])
        choices = W.Select(options=self.get_possible_choices(), layout=dict(width="100%", margin="4px auto"))
        submit = W.Button(description="submit", layout=dict(width="max-content", margin="auto 8px"))
        cancel = W.Button(description="cancel", layout=dict(width="max-content", margin="auto 8px"))
        choice_box = W.VBox(children=(choices, W.HBox(children=(submit, cancel),
                                                      layout=dict(margin="4px auto"))),
                            layout=dict(width="95%", margin="auto"))
        box = W.VBox(children=(header,
                               choice_box,),
                     layout=dict(width=width))

        def show_new_choice(ev):
            box.children = (header, choice_box)
            new_choice.disabled = True

        def add_choice(ev):
            meta = CLUSTERINGS[choices.value]
            self.cfg = meta.config_class()
            new_w = meta.view_func(self.cfg)
            new_w.layout = dict(width="95%")
            box.children = (header, new_w)
            new_choice.disabled = False

        def cancel_new_choice(ev):
            box.children = (*filter(lambda b: b is not choice_box, box.children),)
            new_choice.disabled = False

        submit.on_click(add_choice)
        cancel.on_click(cancel_new_choice)
        new_choice.on_click(show_new_choice)
        self.widget = box

    def get_possible_choices(self):
        return [*CLUSTERINGS.keys()]