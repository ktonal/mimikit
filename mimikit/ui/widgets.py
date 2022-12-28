from ipywidgets import widgets as W
import os

from ..loops.callbacks import tqdm


__all__ = [
    "EnumWidget",
    "pw2_widget",
    "yesno_widget",
    "Labeled",
    "with_tooltip",
    "UploadWidget",
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
