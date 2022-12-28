from ipywidgets import widgets as W
from .. import ui as UI
from ..dataset import DatasetConfig

__all__ = [
    "dataset_view"
]


def dataset_view(cfg: DatasetConfig):
    return UI.ConfigView(
        cfg,
        UI.Param("destination",
                 widget=UI.Labeled(
                     W.HTML(value="<h4>Dataset Filename</h4>", layout=W.Layout(margin="8px auto")),
                     W.Text(value="train.h5", layout=W.Layout(margin="auto")),
                     W.VBox([])
                 )
                 ),
        UI.Param("sources",
                 widget=UI.Labeled(
                     W.HTML("<h4>Select Soundfiles</h4>", layout=W.Layout(margin="8px auto")),
                     UI.SoundFilePicker().widget,
                     W.VBox([])
                 ),
                 compute=lambda conf, ev: tuple(ev.split(", ")))
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                titles=("Dataset Config",), selected_index=0,
                layout=W.Layout(margin="0 auto 0 0", width="500px"))
