from typing import Iterable

from ipywidgets import widgets as W, GridspecLayout
import os

from ..loops.callbacks import tqdm

__all__ = [
    "EnumWidget",
    "pw2_widget",
    "yesno_widget",
    "Labeled",
    "UploadWidget",
]


# TODO:
#  - wavenet/sampleRNN -> all params
#  - refine FilePicker (layout, load/save buttons)
#  - network views from Select
#  - move *_io() to IOSpecs
#  - *_io views (and Select)


def Labeled(
        label, widget, tooltip=None
):
    label_w = W.Label(value=label, tooltip=label)
    if tooltip is not None:
        tltp = W.Button(icon="fa-info", tooltip=tooltip,
                        layout=W.Layout(
                            width="20px",
                            height="12px"),
                        disabled=True,
                        ).add_class("tltp")
        label_w = W.HBox(children=[label_w, tltp], )
    label_w.layout = W.Layout(min_width="max_content", width="auto",
                              overflow='revert')
    container = W.GridBox(children=(label_w, widget),
                          layout=dict(width="auto",
                                      grid_template_columns='1fr 2fr')
                          )
    # container.value = widget.value
    container.observe = widget.observe
    return container


def pw2_widget(
        initial_value,
        min_value=1,
        max_value=2 ** 16,
):
    plus = W.Button(icon="plus", layout=dict(width="auto", overflow='hidden', grid_area='plus'))
    minus = W.Button(icon="minus", layout=dict(width="auto", overflow='hidden', grid_area='minus'))
    value = W.Text(value=str(initial_value), layout=dict(width="auto", overflow='hidden', grid_area='val'))
    plus.on_click(lambda clk: setattr(value, "value", str(min(max_value, int(value.value) * 2))))
    minus.on_click(lambda clk: setattr(value, "value", str(max(min_value, int(value.value) // 2))))
    grid = W.GridBox(children=(minus, value, plus),
                     layout=dict(grid_template_columns='1fr 1fr 1fr',
                                 grid_template_rows='1fr',
                                 grid_template_areas='"minus val plus"'))
    # bind value state to box state
    # grid.value = value.value
    grid.observe = value.observe
    return grid


def yesno_widget(
        initial_value=True,
):
    yes = W.ToggleButton(
        value=initial_value,
        description="yes",
        button_style="success" if initial_value else "",
        layout=dict(width='auto', maring='auto 4px', grid_area='yes')
    )
    no = W.ToggleButton(
        value=not initial_value,
        description="no",
        button_style="" if initial_value else "danger",
        layout=dict(width='auto', maring='auto 4px', grid_area='no')
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
    grid = W.GridBox(children=(yes, no),
                     layout=dict(grid_template_columns='1fr 1fr',
                                 grid_template_rows='1fr',
                                 grid_template_areas='"yes no"'))
    grid.observe = yes.observe
    return grid


def EnumWidget(
        label: str,
        options: Iterable[str],
        value_type=str,
        selected_index=0
):
    options_w = W.GridBox(children=tuple(W.ToggleButton(value=False,
                                                        description=opt,
                                                        tooltip=opt,
                                                        layout=dict(margin='0 4px', width='auto'))
                                         for opt in options),
                          layout=dict(grid_template_columns='1fr ' * len(options),
                                      width='auto', align_self='center'))
    container = Labeled(label, options_w)
    dummy = W.Text(value='')
    if isinstance(selected_index, int):
        value = options_w.children[selected_index].value if value_type is str else \
            value_type(options_w.children[selected_index].value)
        options_w.children[selected_index].value = True
        setattr(options_w.children[selected_index], "button_style", "success")
    else:
        value = selected_index
    container.value = value
    for i, child in enumerate(options_w.children):
        def observer(ev, c=child, index=i):
            val = ev["new"]
            if val and dummy.value != c.description:
                container.selected_index = index
                dummy.value = c.description if value_type is str else value_type(c.description)
                setattr(c, "button_style", "success")
                for other in options_w.children:
                    if other.value and other is not c:
                        other.value = False
                        other.button_style = ""
            elif not val and dummy.value == c.description:
                c.value = True

        child.observe(observer, "value")
    container.observe = dummy.observe
    return container


def UploadWidget(dest="./"):
    def write_uploads(inputs):
        for file in tqdm(inputs["new"], leave=False):
            with open(os.path.join(dest, file.name), "wb") as f:
                f.write(file.content.tobytes())

    upload = W.FileUpload(
        accept='',
        multiple=True,
    )

    upload.observe(write_uploads, names='value')

    return upload
