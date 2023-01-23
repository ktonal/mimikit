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
    label_w = W.Label(value=label)
    if tooltip is not None:
        tltp = W.Button(icon="fa-info", tooltip=tooltip,
                        layout=W.Layout(
                            width="20px",
                            height="12px"),
                        disabled=True,
                        ).add_class("tltp")
        label_w = W.HBox(children=[label_w, tltp], )
    label_w.layout = W.Layout(min_width="max_content", width="auto", overflow='hidden')
    container = W.GridBox(children=(label_w, widget),
                          layout=dict(width="100%", grid_template_columns='1fr 1fr')
                          )
    # container.value = widget.value
    container.observe = widget.observe
    return container


def pw2_widget(
    initial_value,
    min_value=1,
    max_value=2**16,
):
    plus = W.Button(icon="plus", layout=dict(width="auto", overflow='hidden', grid_area='plus'))
    minus = W.Button(icon="minus", layout=dict(width="auto", overflow='hidden', grid_area='minus'))
    value = W.Text(value=str(initial_value), layout=dict(width="auto", overflow='hidden', grid_area='val'))
    plus.on_click(lambda clk: setattr(value, "value", str(min(max_value, int(value.value) * 2))))
    minus.on_click(lambda clk: setattr(value, "value", str(max(min_value, int(value.value) // 2))))
    grid = W.GridBox(children=(plus, value, minus),
                     layout=dict(grid_template_columns='1fr 1fr 1fr',
                                 grid_template_rows='1fr',
                                 grid_template_areas='"plus val minus"'))
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
        layout=dict(width='auto', grid_area='yes')
    )
    no = W.ToggleButton(
        value=not initial_value,
        description="no",
        button_style="" if initial_value else "danger",
        layout=dict(width='auto', grid_area='no')
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
        label,
        options,
        container,
        value_type=str,
        selected_index=0
):
    container.children = (label, *options)
    dummy = W.Text(value='')
    container.value = options[selected_index].value if value_type is str else value_type(options[selected_index].value)
    for i, child in enumerate(options):
        def observer(ev, c=child, index=i):
            val = ev["new"]
            if val and dummy.value != c.description:
                container.selected_index = index
                dummy.value = c.description if value_type is str else value_type(c.description)
                setattr(c, "button_style", "success")
                for other in options:
                    if other.value and other is not c:
                        other.value = False
                        other.button_style = ""
            elif not val and dummy.value == c.description:
                c.value = True

        child.observe(observer, "value")
    options[selected_index].value = True
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
