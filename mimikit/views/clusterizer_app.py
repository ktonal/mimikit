from typing import *
import dataclasses as dtc

import h5mapper
from ipywidgets import widgets as W
import numpy as np
import pandas as pd
from peaksjs_widget import PeaksJSWidget, Segment
import qgrid

from ..config import Config
from ..extract.clusters import *
# from ..extract.label_filter import LabelFilter
from ..features.dataset import DatasetConfig
from ..features.functionals import *
from .clusters import *
from .dataset import dataset_view
from .functionals import *

__all__ = [
    'ComposeTransformWidget',
    'ClusterWidget',
    'ClusterizerApp'
]


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
    "auto-convolve": Meta(AutoConvolve, autoconvolve_view, [Any], False),
    "f0 filter": Meta(F0Filter, f0_filter_view, [MagSpec], False),
    # "top k": Meta(TopKFilter, topk_filter_view, [MagSpec], False),
    "nearest neighbor filter": Meta(NearestNeighborFilter, nearest_neighbor_filter_view, [Any]),
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

    @staticmethod
    def header(box):
        collapse_all = W.Button(description="collapse all",
                                layout=dict(width="max-content", margin="auto 4px auto 2px"))
        expand_all = W.Button(description="expand all",
                              layout=dict(width="max-content", margin="auto auto auto 4px"))
        header = W.HBox(children=[
            W.HTML(value="<h4> Pre Processing Pipeline </h4>", layout=dict(margin="auto")),
            collapse_all, expand_all
        ])

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

        return header

    def __init__(self, compose: Compose = Compose()):
        self.transforms = []
        self.metas = []
        self.new_choice = W.Button(icon="fa-plus", layout=dict(margin="8px auto"))

        self.box = W.VBox(layout=dict(width="50%"))
        header = self.header(self.box)
        self.box.children = (header,)

        self.choices = W.Select(options=self.get_possible_choices(), layout=dict(width="100%", margin="4px auto"))
        self.submit = W.Button(description="submit", layout=dict(width="max-content", margin="auto 8px"))
        self.cancel = W.Button(description="cancel", layout=dict(width="max-content", margin="auto 8px"))
        self.choice_box = W.VBox(children=(self.choices, W.HBox(children=(self.submit, self.cancel),
                                                                layout=dict(margin="4px auto"))),
                                 layout=dict(width="calc(100% - 54px)", margin="auto 0 auto 54px"))

        self.submit.on_click(self.add_choice)
        self.cancel.on_click(self.cancel_new_choice)
        self.new_choice.on_click(self.show_new_choice)
        self.widget = self.box

        if any(compose.functionals):
            children = []
            for cfg in compose.functionals:
                meta_key = self.get_transforms_key_for(cfg)
                meta, cfg, remove_w, func_view = self.new_transform_view_for(TRANSFORMS[meta_key], cfg)
                self.metas.append(meta)
                self.transforms.append(cfg)
                children += [func_view]
            self.box.children = (header, *children, self.new_choice)
        else:
            self.choices.value = "magspec"
            self.submit.click()
            # can not remove magspec
            self.box.children[1].children[0].disabled = True

    @property
    def magspec_cfg(self):
        return self.transforms[0]

    def show_new_choice(self, ev):
        self.box.children = (*filter(lambda b: b is not self.new_choice, self.box.children), self.choice_box)
        self.new_choice.disabled = True

    def add_choice(self, ev):
        meta, cfg, remove_w, hbox = self.new_transform_view_for(TRANSFORMS[self.choices.value])
        self.metas.append(meta)
        self.transforms.append(cfg)
        self.choices.options = self.get_possible_choices()

        def remove_cb(ev):
            keep = [0]  # always keep magspec
            for i, (t, m) in enumerate(zip(self.transforms[1:], self.metas[1:]), 1):
                is_el = t is cfg
                requires_t = type(cfg) in m.requires
                if not is_el and not requires_t:
                    keep += [i]
            self.transforms = [self.transforms[i] for i in keep]
            self.box.children = (self.box.children[0],) + tuple(self.box.children[i + 1] for i in keep) + (
                self.box.children[-1],)
            self.choices.options = self.get_possible_choices()

        remove_w.on_click(remove_cb)
        self.box.children = (*filter(lambda b: b is not self.choice_box, self.box.children), hbox, self.new_choice)
        self.new_choice.disabled = False

    def cancel_new_choice(self, ev):
        self.box.children = (*filter(lambda b: b is not self.choice_box, self.box.children), self.new_choice)
        self.new_choice.disabled = False

    def get_possible_choices(self):
        options = []
        ts = [*map(type, self.transforms)]
        for k, meta in TRANSFORMS.items():
            if meta.can_be_added(ts):
                options += [k]
        return options

    def new_transform_view_for(self, meta: Meta, cfg: Optional[Config] = None):
        if cfg is None:
            cfg = meta.config_class()
        new_w = meta.view_func(cfg)
        new_w.layout = dict(margin="auto", width='100%')
        remove_w = W.Button(icon="fa-trash", layout=dict(width="50px", margin="auto 2px"))
        hbox = W.HBox(children=(remove_w, new_w), layout=dict(margin="0 4px 4px 4px"))
        return meta, cfg, remove_w, hbox

    @staticmethod
    def get_transforms_key_for(config):
        return next(k for k, v in TRANSFORMS.items() if v.config_class is type(config))

    @staticmethod
    def display(cfg: Compose):
        w = []
        for func in cfg.functionals:
            tp = type(func)
            key = next(k for k, v in TRANSFORMS.items() if v.config_class is tp)
            meta = TRANSFORMS[key]
            w += [meta.view_func(func)]
        box = W.VBox(layout=dict(width="50%"))
        header = ComposeTransformWidget.header(box)
        box.children = (header, *w)
        return box


class ClusterWidget:
    def __init__(self):
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
                     layout=dict(width='50%'))

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

    @staticmethod
    def get_possible_choices():
        return [*CLUSTERINGS.keys()]

    @staticmethod
    def display(cfg):
        tp = type(cfg)
        key = next(k for k, v in CLUSTERINGS.items() if v.config_class is tp)
        box = W.VBox(children=[
            W.HTML(value="<h4> Clustering Algo </h4>", layout=dict(margin="auto")),
            CLUSTERINGS[key].view_func(cfg),
        ], layout=dict(width="50%"))
        return box


class ClusterizerApp:

    def __init__(self):
        self.dataset_cfg = DatasetConfig()
        self.dataset_widget = dataset_view(self.dataset_cfg)
        self.labels_grid = None
        self.bounced_container = W.VBox(layout=dict(width="100%"))
        self.labels_widget = W.VBox(layout=dict(max_width="90vw", margin="auto"))
        self.feature_name = ''
        self.selected_labels = set()
        self.pre_pipeline = None
        self.pre_pipeline_widget = None
        self.clusters = None
        self.clusters_widget = None
        self.main_waveform = None
        self.out = W.Output()
        self.clustering_widget = W.Tab(children=(
            W.HTML("<div> </div>", layout=dict(height="200px")),
            W.HTML("<div> </div>", layout=dict(height="200px"))
        ),
            layout=dict(max_width="1000px", margin="auto"))
        self.clustering_widget.set_title(0, 'Create new clustering')
        self.clustering_widget.set_title(1, 'Load clustering')

        def reset_clustering_view(ev):
            if ev["new"] == 0:
                self.build_new_clustering_view()
            else:
                self.build_load_clustering_view(self.db)

        self.clustering_widget.observe(reset_clustering_view, "selected_index")
        self.dataset_widget.on_created(lambda ev: self.load_dataset())
        self.dataset_widget.on_loaded(lambda ev: self.load_dataset())

    def load_dataset(self):
        self.labels_widget.children = tuple()
        self.build_main_waveform()
        self.build_new_clustering_view()
        self.build_load_clustering_view(self.db)

    def build_main_waveform(self):
        self.main_waveform = PeaksJSWidget(
            array=self.db.signal[:], sr=self.sr, id_count=0,
            with_save_button=True, with_play_button=True,
            layout=dict(margin="auto", max_width="1500px", width="100%")
        )

    def build_new_clustering_view(self):
        self.pre_pipeline = ComposeTransformWidget()
        self.pre_pipeline_widget = self.pre_pipeline.widget
        self.clusters = ClusterWidget()
        self.clusters_widget = self.clusters.widget
        self.save_as = W.HBox(children=(
            W.Label(value='Save clustering as: '), W.Text(value="labels")),
            layout=dict(margin="auto", width="max-content"))
        compute = W.HBox(children=(W.Button(description="compute"),),
                         layout=dict(margin="auto", width="max-content"))
        compute.children[0].on_click(self.on_submit)

        self.clustering_widget.children = (
            W.VBox(children=(
                W.HBox(children=(self.pre_pipeline_widget, self.clusters_widget),
                       layout=dict(align_items='baseline')),
                self.save_as, compute, self.out
            )),
            *self.clustering_widget.children[1:]
        )

    def build_load_clustering_view(self, db):
        proxies = [k for k, v in db.__dict__.items()
                   if isinstance(v, h5mapper.Proxy) and not k.startswith("__") and k != "signal"
                   ]
        self.results_buttons = W.ToggleButtons(options=proxies, index=None,
                                               layout=dict(margin="12px 0"))

        def callback(ev):
            self.load_result(ev["new"])
            self.feature_name = ev["new"]
            self.selected_labels = set()
            self.labels_widget.children = self.label_view()

        self.results_buttons.observe(callback, "value")
        self.clustering_widget.children = (
            self.clustering_widget.children[0],
            W.VBox(children=(W.Label(value="load clustering: "), self.results_buttons))
        )

    def load_result(self, key: str):
        cfg = Config.deserialize(getattr(self.db, key).attrs["config"])
        compose = Compose(*cfg.functionals[:-1])
        clustering = cfg.functionals[-1]
        self.pre_pipeline = ComposeTransformWidget(compose)
        self.pre_pipeline_widget = self.pre_pipeline.widget
        self.clustering_widget.children = (
            self.clustering_widget.children[0],
            W.VBox(children=(
                W.VBox(children=(W.Label(value="load clustering: "), self.results_buttons)),
                W.HBox(children=(self.pre_pipeline_widget, ClusterWidget.display(clustering)),
                       layout=dict(align_items='baseline'))
            ))
        )

    def on_submit(self, ev):
        db = self.db
        self.feature_name = self.save_as.children[1].value
        if self.clusters.cfg is None:
            with self.out:
                raise ValueError("Please select a clustering algo before clicking on 'compute'")
            return
        pipeline = Compose(
            *self.pre_pipeline.transforms, self.clusters.cfg,
            # Interpolate(mode="previous", length=db.signal.shape[0])
        )
        if self.feature_name in db.handle():
            db.handle().pop(self.feature_name)
            db.flush()
        self.out.clear_output()
        with self.out:
            db.signal.compute({
                self.feature_name: pipeline
            }, parallelism='none')
            feat = getattr(db, self.feature_name)
            feat.attrs["config"] = pipeline.serialize()
            db.flush()
            db.close()
        self.main_waveform.segments = []
        self.clustering_widget.selected_index = 1
        self.build_load_clustering_view(self.db)
        self.results_buttons.value = self.feature_name

    @property
    def db(self):
        return self.dataset_cfg.get(mode="r+")

    @property
    def sr(self):
        return self.db.config.extractors[0].functional.functionals[0].sr

    @property
    def hop_length(self):
        return self.magspec_cfg.hop_length

    @property
    def magspec_cfg(self):
        return self.pre_pipeline.magspec_cfg

    @property
    def signal(self):
        return self.db.signal

    @property
    def labels(self):
        return getattr(self.db, self.feature_name)

    @property
    def segments_from_clustering(self):
        """segments inferred from the raw labels of the current clustering"""
        sr = self.sr
        lbl = self.labels[:]
        splits = (lbl[1:] - lbl[:-1]) != 0
        time_idx = splits.nonzero()[0] + 1
        starts = np.r_[0, time_idx]
        ends = np.r_[time_idx, lbl.shape[0]]
        cluster_idx = lbl[starts]
        return [
            Segment(t, tp1, id=i, labelText=str(c)).dict()
            for i, (t, tp1, c) in enumerate(
                zip((self.hop_length * starts) / sr,
                    ((self.hop_length * ends) / sr),
                    cluster_idx))
        ]

    @property
    def segments(self):
        """segments as currently edited by the user"""
        return sorted(self.main_waveform.segments, key=lambda s: s['startTime'])

    def select(self, *labels: int):
        labels_str = {*map(str, labels)}
        for hbox in self.labels_grid.children:
            btn = hbox.children[0]
            if btn.description in labels_str and not btn.value:
                btn.value = True
        return self

    def unselect(self, *labels: int):
        labels_str = {*map(str, labels)}
        for hbox in self.labels_grid.children:
            btn = hbox.children[0]
            if btn.description in labels_str and btn.value:
                btn.value = False
        return self

    def bounce_selected_labels(self):
        """bounce the segments for the selected labels ignoring user's updates"""
        fft = self.magspec_cfg(self.signal[:])
        mask = np.zeros((fft.shape[0],), dtype=bool)
        labels = self.labels[:]
        for label in self.selected_labels:
            mask += labels == int(label)

        filtered = fft[mask]
        return self.magspec_cfg.inv(filtered)

    def bounce_segments(self):
        """bounce the segments for the selected labels as updated by the user"""
        segments = self.segments
        fft = self.magspec_cfg(self.signal[:])
        sr, hop = self.sr, self.hop_length

        def t2f(t): return int(round((t * sr) / hop))

        filtered = np.concatenate([fft[slice(
            t2f(s['startTime']),
            t2f(s['endTime']))]  # segments_from_clustering already contains `+ 1`
                                   for s in segments])
        return self.magspec_cfg.inv(filtered)

    def segments_df_and_labels_set(self):
        df = pd.DataFrame.from_dict(self.segments_from_clustering)
        df.set_index("id", drop=True, inplace=True)
        label_set = [*map(str, sorted(map(int, pd.unique(df.labelText))))]
        return df, label_set

    def label_view(self):
        self.main_waveform.segments = []
        df, label_set = self.segments_df_and_labels_set()
        df.to_dict()
        self.main_waveform.id_count = len(df)
        empty = pd.DataFrame([])
        empty.index.name = "id"
        g = qgrid.show_grid(empty,
                            grid_options=dict(maxVisibleRows=10))

        self.labels_grid = W.GridBox(layout=dict(max_height='400px',
                                                 margin="16px auto",
                                                 grid_template_columns="1fr " * 10,
                                                 grid_auto_rows="max-content",
                                                 grid_gap="8px",
                                                 overflow="scroll"))
        labels_w = []

        for label_str in label_set:
            btn = W.ToggleButton(value=False,
                                 description=label_str,
                                 layout=dict(width="100px",
                                             min_height="30px",
                                             padding="2px",
                                             margin="auto",
                                             overflow="clip !important"
                                             ))

            def on_click(ev, widget=btn, index=label_str):
                w = self.main_waveform
                if ev["new"]:
                    w.segments = [
                        *w.segments, *df[df.labelText == index].reset_index().T.to_dict().values()
                    ]
                    widget.button_style = "success"
                    self.selected_labels.add(index)
                else:
                    w.segments = [
                        s for s in w.segments if s["labelText"] != index
                    ]
                    widget.button_style = ""
                    self.selected_labels.remove(index)
                if w.segments:
                    g.df = pd.DataFrame.from_dict(w.segments).sort_values(by="startTime").set_index("id", drop=True)
                else:
                    g.df = pd.DataFrame([])
                    g.df.index.name = "id"

            btn.observe(on_click, "value")
            labels_w += [W.HBox(children=(btn,))]

        self.labels_grid.children = tuple(labels_w)

        def on_new_segment(wdg, seg):
            w = self.main_waveform
            new_seg = Segment(**seg).dict()
            if w.segments:
                g.add_row(row=[*new_seg.items()])
                # i = {**new_seg}.pop("id")
                # df.loc[i] = {**new_seg}
            else:
                g.df = pd.DataFrame.from_dict([new_seg]).set_index("id", drop=True)

        def on_edit_segment(wdg, seg):
            seg = Segment(**seg).dict()
            for k, v in seg.items():
                if k == "id": continue
                g.edit_cell(seg["id"], k, v)
                # df.loc[seg["id"], k] = v
            g.change_selection([seg["id"]])

        def on_remove_segment(wdg, seg):
            g.remove_row([seg["id"]])
            # df.drop(seg["id"], inplace=True)

        def segments_changed(ev):
            pass
            # print("segments changed")

        self.main_waveform.observe(segments_changed, "segments")
        self.main_waveform.on_new_segment(on_new_segment)
        self.main_waveform.on_new_segment(PeaksJSWidget.add_segment)
        self.main_waveform.on_edit_segment(on_edit_segment)
        self.main_waveform.on_edit_segment(PeaksJSWidget.edit_segment)
        self.main_waveform.on_remove_segment(on_remove_segment)
        self.main_waveform.on_remove_segment(PeaksJSWidget.remove_segment)

        # g.on("selection_changed", on_selection_changed)
        # g.on("cell_edited", on_edited_cell)
        # g.on("filter_changed", lambda ev, qg: print(ev))
        # g.on("row_removed", on_row_removed)

        def on_bounce(ev):
            title = ", ".join(map(str, sorted(map(int, self.selected_labels))))
            title = W.HTML(f"<h4>Clustering: '{self.feature_name}' - Labels: {title}</h4>")
            bnc = self.bounce_segments()
            peaks = PeaksJSWidget(
                array=bnc, sr=self.sr, id_count=0,
                with_save_button=True, with_play_button=True,

            )
            remove_w = W.Button(icon="fa-trash", layout=dict(width="50px"))
            new_waveform = W.VBox(children=(W.HBox(children=(remove_w, title,)), peaks,))

            def on_remove(ev):
                self.bounced_container.children = tuple(
                    el for el in self.bounced_container.children if el is not new_waveform
                )

            remove_w.on_click(on_remove)
            self.bounced_container.children = (
                new_waveform,
                *self.bounced_container.children,
            )

        bounce = W.Button(description="Bounce Selection")
        bounce.on_click(on_bounce)

        def on_reset_selected_labels(ev):
            for w in self.labels_grid.children:
                if w.children[0].value:
                    w.children[0].value = False
            self.selected_labels = set()
            self.main_waveform.segments = []

        reset_label_button = W.Button(description="Reset Selection")
        reset_label_button.on_click(on_reset_selected_labels)

        return (
            self.main_waveform,
            W.HTML("<h4>Select Label(s): </h4>"),
            W.HBox(children=(reset_label_button, bounce),
                   layout=dict(margin="8px auto",
                               )),
            self.labels_grid,
            W.HTML("<h4>Selected Labels Segments Table: </h4>"),
            g,
        )
