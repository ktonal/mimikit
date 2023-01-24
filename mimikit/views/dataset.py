from ipywidgets import widgets as W
from .. import ui as UI
from ..features.functionals import Compose, FileToSignal, RemoveDC, Normalize
from ..features.extractor import Extractor
from ..features.dataset import DatasetConfig

__all__ = [
    "dataset_view"
]


def dataset_view(cfg: DatasetConfig):
    title = W.HTML("<h4>Select Soundfiles</h4>", layout=dict(margin='0 0 0 8px'))
    picker = UI.SoundFilePicker()
    picker_w = picker.widget
    save_as_txt = W.Text(value=cfg.filename, description='Save as: ',
                         layout=dict(width='75%', margin='auto 0 0 8px'))
    create = W.Button(description="Create", layout=dict(width='75%', margin='auto 0 0 8px'))

    sample_rate = W.IntText(value=16000, description="Sample Rate: ",
                            layout=dict(
                                width='max-content',
                                margin='auto 0 0 8px')
                            )
    out = W.Output()
    new_ds_container = W.AppLayout(
        header=title,
        center=picker_w,
        footer=W.VBox(children=(sample_rate, save_as_txt, create)),
        pane_widths=('0fr', '3fr', '0fr'),
        pane_heights=("32px", "230px", "125px")
    )

    def create_ds(ev):
        cfg.sources = tuple(picker.selected)
        cfg.extractors = (Extractor(name='signal',
                                    functional=Compose(
                                        FileToSignal(sample_rate.value),
                                        RemoveDC(),
                                        Normalize()
                                    )),)
        new_ds_container.footer = None
        new_ds_container.header = None
        new_ds_container.center = out
        with out:
            db = cfg.create(mode='w')
            print("Extracted:\n\n", *(f"\t- {k}\n" for k in db.index))
            print("Dataset file summary: ")
            db.info()

    create.on_click(create_ds)

    ds_picker = UI.DatasetPicker()
    ds_picker_w = ds_picker.widget
    load_ds = W.Button(description="Load")
    load_out = W.Output()
    load_ds_container = W.VBox(children=[
        W.HTML(value="<h4>Select Dataset File</h4>"),
        ds_picker_w,
        load_ds
    ])

    def load_cb(ev):
        cfg.filename = ds_picker.selected
        db = cfg.get(mode='r')
        load_ds_container.children = (load_out, )
        with load_out:
            print("Loaded", cfg.filename)
            print("Containing:\n\n", *(f"\t- {k}\n" for k in db.index))
            print("Dataset file summary: ")
            db.info()

    load_ds.on_click(load_cb)

    tabs = W.Tab(children=(new_ds_container, load_ds_container))
    tabs.set_title(0, "New Dataset from Soundfiles")
    tabs.set_title(1, "Load Dataset File from Disk")
    top = W.Accordion(children=(tabs, ))
    top.set_title(0, "Dataset")
    return top

