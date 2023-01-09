from ipywidgets import widgets as W
from .. import ui as UI
from ..features.dataset import DatasetConfig

__all__ = [
    "dataset_view"
]


def dataset_view(cfg: DatasetConfig):
    view = UI.ConfigView(
        cfg,
        UI.Param("sources",
                 widget=UI.Labeled(
                     W.HTML("<h4>Select Soundfiles</h4>", layout=W.Layout(margin="4px")),
                     UI.SoundFilePicker().widget,
                     W.VBox([])
                 ),
                 compute=lambda conf, ev: tuple(ev.split(", "))),
        UI.Param("destination",
                 widget=UI.Labeled(
                     W.HTML(value="<b>Output Filename: </b>",
                            layout=W.Layout(min_width="max-content", margin="0 4px")),
                     W.Text(value="train.h5", layout=W.Layout(width="100%")),
                     W.HBox([], layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 )
                 ),
    ).as_widget(lambda children, **kwargs: W.VBox(children=children, **kwargs),
                layout=W.Layout(margin="0 auto 0 0", width="100%"))
    return view
