from ipywidgets import widgets as W
from .. import ui as UI
from ..features.functionals import Compose, FileToSignal, RemoveDC, Normalize
from ..features.extractor import Extractor
from ..features.dataset import DatasetConfig

__all__ = [
    "dataset_view"
]


def dataset_view(cfg: DatasetConfig):

    out = W.Output(layout=dict(margin='32px 0'))
    title = W.HTML("<h4>Select Soundfiles</h4>", layout=dict(margin='0 0 0 8px'))
    picker = UI.SoundFilePicker()
    picker_w = picker.widget
    save_as_txt = W.Text(value=cfg.filename, description='Save as: ',
                         layout=dict(width='75%', margin='4px 0 4px 8px'))
    create = W.Button(description="Create", layout=dict(width='75%', margin='4px 0 4px 8px'))

    sample_rate = W.IntText(value=16000, description="Sample Rate: ",
                            layout=dict(
                                width='max-content',
                                margin='4px 0 4px 8px')
                            )
    new_ds_container = W.AppLayout(
        header=title,
        center=picker_w,
        footer=W.VBox(children=(sample_rate, save_as_txt, create),
                      layout=dict(width='75%')),
        # pane_widths=('0fr', '3fr', '0fr'),
        pane_heights=("40px", "250px", "112px")
    )

    def create_ds(ev, callback=None):
        cfg.sources = tuple(picker.selected)
        cfg.extractors = (Extractor(name='signal',
                                    functional=Compose(
                                        FileToSignal(sample_rate.value),
                                        RemoveDC(),
                                        Normalize()
                                    )),)
        cfg.filename = save_as_txt.value
        out.clear_output()
        with out:
            db = cfg.create(mode='w')
            print("Extracted:\n\n", *(f"\t- {k}\n" for k in db.index))
        if callback is not None:
            callback(db)

    create.on_click(create_ds)

    ds_picker = UI.DatasetPicker()
    ds_picker_w = ds_picker.widget
    load_ds = W.Button(description="Load")
    load_ds_container = W.VBox(children=[
        W.HTML(value="<h4>Select Dataset File</h4>"),
        ds_picker_w,
        load_ds
    ])

    def load_cb(ev, callback=None):
        cfg.filename = ds_picker.selected
        db = cfg.get(mode='r')
        out.clear_output()
        with out:
            print("Loaded", cfg.filename)
            print("Containing:\n\n", *(f"\t- {k}\n" for k in db.index))
        if callback is not None:
            callback(db)

    load_ds.on_click(load_cb)

    tabs = W.Tab(children=(new_ds_container, load_ds_container))
    tabs.set_title(0, "New Dataset from Soundfiles")
    tabs.set_title(1, "Load Dataset File from Disk")

    class DatasetView(W.Accordion):
        def __init__(self, *children, **kwargs):
            super(DatasetView, self).__init__(children=children, **kwargs)
            self.create = create
            self.load_ds = load_ds

        def on_created(self, callback):
            self.create.on_click(callback)

        def on_loaded(self, callback):
            self.load_ds.on_click(callback)

    top = DatasetView(W.VBox(children=(tabs, out)),
                      layout=dict(max_width="1000px", margin="auto"))
    top.set_title(0, "Dataset")
    return top

