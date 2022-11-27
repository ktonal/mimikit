from typing import List, Dict
from ipywidgets import widgets as W
import dataclasses as dtc
import os
import h5mapper as h5m

from .ipyfilechooser.filechooser import FileChooser

__all__ = [
    "pw2_widget",
    "yesno_widget",
    "FileWidget"
]


@dtc.dataclass
class FileWidget:
    root: str = os.path.expanduser("~")
    files: Dict[str, bool] = dtc.field(default_factory=dict)
    regex: str = h5m.Sound.__re__

    chooser_layout: W.Layout = W.Layout(width="100%")
    file_layout: W.Layout = W.Layout(width="auto")
    files_grid_layout: W.Layout = W.Layout(grid_template_columns='1fr 1fr 1fr 1fr 1fr',
                                           max_height="250px", overflow="scroll")

    def new_chooser(self):
        chsr = FileChooser(self.root, show_only_dirs=True, layout=self.chooser_layout)
        chsr._dircontent.rows = 12
        # chsr._dircontent.layout = W.Layout(width='auto', grid_area='dircontent', grid_template_columns="1fr 1fr 1fr")
        chsr.register_callback(lambda chs: self.add_dir(chs.selected_path))
        return chsr

    def __post_init__(self):
        plus = W.Button(icon="plus", layout=W.Layout(width="95%"),
                        description="Add directory")
        plus.on_click(self.add_chooser)
        self.choosers = W.VBox([], layout=W.Layout(margin="10px"))
        self.files_btn = W.GridBox([], layout=self.files_grid_layout)
        self.widget = W.VBox([plus, self.choosers])
        self.widget.observe = self.observe
        self._callbacks = []

    @property
    def value(self):
        return [f for f, b in self.files.items() if b]

    @value.setter
    def value(self, files):
        self.files = {f: True for f in files}

    def add_chooser(self, clk):
        self.choosers.children += (self.new_chooser(),)
        return self

    def add_dir(self, directory):
        self.walk_dir(directory)
        if len(self.widget.children) == 2:
            slct_all = W.Checkbox(value=False, description="Select All", indent=False, layout=W.Layout(width="50%"))

            def select_all(ev):
                for b in self.files_btn.children:
                    b.value = ev["new"]

            slct_all.observe(select_all, "value")
            self.widget.children += (slct_all, W.Label(value="click the files to include"), self.files_btn,)

    def walk_dir(self, directory):
        for p in h5m.FileWalker(self.regex, directory):
            self.files[p] = False
            wdg = W.ToggleButton(description=os.path.split(p)[1], value=False, layout=self.file_layout)

            def observer(ev, path=p, w=wdg):
                self.files.update({path: ev["new"]})
                w.button_style = "success" if ev["new"] else ""
                self.callback()

            wdg.observe(observer, "value")
            self.files_btn.children += (wdg,)
        return self

    def observe(self, callback, _):
        self._callbacks += [callback]
        return self

    def callback(self):
        files = self.value
        for cb in self._callbacks:
            cb({"new": files})
        return self

