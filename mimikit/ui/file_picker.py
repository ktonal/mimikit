import os
import re
from ipywidgets import widgets as W
from functools import partial
from h5mapper import Sound

__all__ = [
    "FilePicker",
    "SoundFilePicker",
    "CheckpointPicker"
]


class FilePicker:
    def __init__(self,
                 root=os.getcwd(),
                 multiple=True,
                 show_hidden=False,
                 pattern=".*",
                 n_columns=4):
        self.root = root
        self.n_columns = n_columns
        self.show_hidden = show_hidden
        self.pattern = re.compile(pattern)
        self.multiple = multiple
        self.widget = W.VBox([
            W.Text(value=self.root,
                   description="current dir:",
                   disabled=True,
                   layout=W.Layout(width="100%", margin="0 0 12px 0"))
                .add_class("selected"),
            W.GridBox(layout=W.Layout(grid_template_columns="1fr " *
                                                            self.n_columns,
                                      grid_auto_rows="min-content",
                                      width="98%",
                                      height="200px")),
            W.Text(description="selected:",
                   disabled=True,
                   layout=W.Layout(width="100%",
                                   margin="8px 0 0 0"))
                .add_class("selected")
        ],
            layout=W.Layout(width="100%", ))
        self.widget.observe = self.widget.children[-1].observe
        self.widget.value = self.widget.children[-1].value
        self.selected = set() if self.multiple else None
        self.update()

    def update(self):
        self.widget.children[1].children = \
            [
                W.Button(description='\U0001F4C1 ..', layout=dict(width="auto"))
            ] + \
            [W.Button(description=('\U0001F4C1 ' if os.path.isdir(
                os.path.join(self.root, path)) else "") + path,
                      disabled=self.disabled(path), tooltip=path,
                      layout=dict(width="auto"))
                 .add_class("picker-button")
             for path in sorted(os.listdir(self.root)) if self.show_path(path)
             ]

        for button in self.widget.children[1].children:
            button.add_class("picker-button")
            #             print(button.description, self.selected)
            if button.tooltip is not None and os.path.join(
                    self.root, button.tooltip) in self.selected:
                button.add_class("selected-button")
            button.on_click(self.click_path)

    def show_path(self, path):
        show = True
        is_dir = os.path.isdir(os.path.join(self.root, path))
        if path[0] == '.' and not self.show_hidden:
            show = False
        if not bool(re.search(self.pattern, path)) and not is_dir and show:
            show = True
        return show

    def disabled(self, path):
        return not bool(re.search(self.pattern, path)) and not os.path.isdir(
            os.path.join(self.root, path))

    def click_path(self, button):
        desc = button.description
        if desc.startswith('\U0001F4C1 '):
            self.root = os.path.abspath(
                os.path.join(self.root, desc.strip('\U0001F4C1 ')))
            self.widget.children[0].value = self.root
            self.update()
        else:
            desc = os.path.join(self.root, desc)
            if self.multiple:
                if desc in self.selected:
                    self.selected.remove(desc)
                    button.remove_class("selected-button")
                else:
                    self.selected.add(desc)
                    button.add_class("selected-button")
            else:
                if self.selected == desc:
                    button.remove_class("selected-button")
                    self.selected = None
                else:
                    button.remove_class("selected-button")
                    self.selected = desc
        self.widget.children[-1].value = os.path.split(self.selected)[-1] \
            if not self.multiple else ", ".join([os.path.split(p)[-1] for p in self.selected])


SoundFilePicker = partial(FilePicker, pattern=Sound.__re__)
CheckpointPicker = partial(FilePicker, pattern=r".*\.h5$")
