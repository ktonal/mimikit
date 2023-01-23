from ipywidgets import widgets as W
from .. import ui as UI
from ..features.dataset import DatasetConfig

__all__ = [
    "dataset_view"
]


def dataset_view(cfg: DatasetConfig):

    title = W.HTML("<h4>Select Soundfiles</h4>")
    picker = UI.SoundFilePicker().widget
    save_as = W.HTML("<b>Save as: </b>")
    save_as_txt = W.Text(value=cfg.filename)
    save_as_btn = W.Button(description="Save")
    selected = W.VBox(disabled=True, layout=dict(height="255px", overflow="scroll"))
    picker.observe(lambda ev: setattr(selected, 'children',
                                      tuple(W.HTML(f"<li>{s}</li>", layout=dict(width='auto'))
                                            .add_class('selected')
                                            for s in ev["new"].split("<$>"))), 'value')
    # selected.value = picker.value
    container = W.AppLayout(
        header=title,
        right_sidebar=selected,
        center=picker,
        footer=W.HBox(children=(save_as, save_as_txt, save_as_btn),
                      layout=dict(width='100%'))
    )
    return container
