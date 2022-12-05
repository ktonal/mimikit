from ipywidgets import widgets as W
import os
import re


__all__ = [
    "EnumWidget",
    "pw2_widget",
    "yesno_widget",
    "Labeled",
    "with_tooltip",
    "FilePicker"
]


def Labeled(
        label, value, container
):
    container.children = (label, value)
    container.value = value.value
    container.observe = value.observe
    return container


def EnumWidget(
        label,
        options,
        container,
        value_type=str,
        selected_index=0
):
    container.children = (label, *options)
    container.value = options[selected_index].value if value_type is str else value_type(options[selected_index].value)
    for i, child in enumerate(options):
        def observer(ev, c=child, index=i):
            val = ev["new"]
            if val:
                container.selected_index = index
                container.value = c.description if value_type is str else value_type(c.description)
                setattr(c, "button_style", "success")
                for other in options:
                    if other.value and other is not c:
                        other.value = False
            else:
                c.button_style = ""

        child.observe(observer, "value")
    options[selected_index].value = True
    return container


def pw2_widget(
    label,
    value,
    plus,
    minus,
    container,
    min_value=1,
    max_value=2**16,
):
    plus.on_click(lambda clk: setattr(value, "value", str(min(max_value, int(value.value) * 2))))
    minus.on_click(lambda clk: setattr(value, "value", str(max(min_value, int(value.value) // 2))))
    container.children = [label, minus, value, plus]
    # bind value state to box state
    container.observe = value.observe
    return container


def yesno_widget(
    label,
    container,
    initial_value=True,
    buttons_layout={}
):
    yes = W.ToggleButton(
        value=initial_value,
        description="yes",
        button_style="success" if initial_value else "",
        layout=buttons_layout
    )
    no = W.ToggleButton(
        value=not initial_value,
        description="no",
        button_style="" if initial_value else "danger",
        layout=buttons_layout
    )

    def toggle_yes(ev):
        v = ev["new"]
        if v:
            setattr(yes, "button_style", "success")
            setattr(no, "button_style", "")
            setattr(no, "value", False)

    def toggle_no(ev):
        v = ev["new"]
        if v:
            setattr(no, "button_style", "danger")
            setattr(yes, "button_style", "")
            setattr(yes, "value", False)

    yes.observe(toggle_yes, "value")
    no.observe(toggle_no, "value")

    container.children = [label, yes, no]
    container.observe = yes.observe
    return container


def with_tooltip(widget, tooltip):
    tltp = W.Button(icon="fa-info", tooltip=tooltip,
                    layout=W.Layout(
                        width="20px",
                        height="12px"),
                    disabled=True,
                   ).add_class("tltp")
    return W.HBox([widget, tltp], layout=W.Layout(min_width="max_content",))


class FilePicker:

    def __init__(self,
                 root=os.getcwd(),
                 multiple=True,
                 show_hidden=False,
                 pattern=".*",
                 n_columns=4
                 ):
        self.root = root
        self.n_columns = n_columns
        self.show_hidden = show_hidden
        self.pattern = re.compile(pattern)
        self.multiple = multiple
        self.widget = W.VBox([
            W.Text(value=self.root, disabled=True,
                   layout=W.Layout(width="100%", margin="0 0 12px 0")),
            W.GridBox(
                layout=W.Layout(grid_template_columns="1fr " * self.n_columns,
                                grid_auto_rows="min-content",
                                width="98%", height="200px")),
            W.Text(disabled=True,
                   layout=W.Layout(width="100%", margin="8px 0 0 0")).add_class("selected")],
            layout=W.Layout(width="100%", ))
        self.selected = set() if self.multiple else None
        self.update()

    def update(self):
        self.widget.children[1].children = [W.Button(description='\U0001F4C1 ..').add_class("not-a-button")] + \
                                           [W.Button(description=('\U0001F4C1 ' if os.path.isdir(
                                               os.path.join(self.root, path)) else "") + path,
                                                     disabled=False, tooltip=path, layout=dict(width="auto")).add_class(
                                               "not-a-button")
                                            for path in sorted(os.listdir(self.root)) if self.show_path(path)]

        for button in self.widget.children[1].children:
            button.on_click(self.click_path)

    def show_path(self, path):
        show = True
        is_dir = os.path.isdir(os.path.join(self.root, path))
        if path[0] == '.' and not self.show_hidden:
            show = False
        if not self.pattern.match(path) and not is_dir:
            show = False
        return show

    def click_path(self, button):
        desc = button.description
        if desc.startswith('\U0001F4C1 '):
            self.root = os.path.abspath(os.path.join(self.root, desc.strip('\U0001F4C1 ')))
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
