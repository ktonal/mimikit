import ipywidgets as W
from .. import ui as UI
from ..networks.sample_rnn_v2 import SampleRNN

__all__ = [
    "sample_rnn_view"
]


def sample_rnn_view(cfg: SampleRNN.Config):
    label_layout = W.Layout(min_width="max-content", margin="0 0 0 auto")
    param_layout = W.Layout(width="100%", margin="8px 0 8px 0")

    view = UI.ConfigView(
        cfg,
        UI.Param(
            name='frame_sizes',
            widget=UI.Labeled(
                W.Label(value="Frame Sizes",
                        layout=label_layout),
                W.Text(value=str(cfg.frame_sizes)[1:-1], layout=W.Layout(width="75%", )),
                W.HBox(layout=param_layout), ),
            compute=lambda c, v: tuple(map(int, (s for s in v.split(",") if s not in ("", " "))))
        ),
        UI.Param(
            name='hidden_dim',
            widget=UI.pw2_widget(
                W.Label(value="Hidden Dim: ",
                        layout=label_layout, ),
                W.Text(value=str(cfg.hidden_dim),
                       layout=W.Layout(width="25%"), disabled=False),
                W.Button(icon="plus", layout=W.Layout(width="25%")),
                W.Button(icon="minus", layout=W.Layout(width="25%")),
                W.HBox(layout=param_layout)),
            compute=lambda c, v: int(v)
        ),
        UI.Param(
            name="rnn_class",
            widget=UI.EnumWidget(
                W.Label(value="Type of RNN: ", layout=label_layout),
                [W.ToggleButton(description="LSTM"), W.ToggleButton(description="RNN"), W.ToggleButton(description="GRU")],
                W.HBox()
            ),
            compute=lambda c, v: v.lower()
        ),
        UI.Param(
            name="n_rnn",
            widget=UI.Labeled(
                W.Label("Num of RNN: ", layout=label_layout),
                W.IntText(value=cfg.n_rnn,
                          layout={"width": "100px", }),
                W.HBox(layout=param_layout)
            )
        ),
        UI.Param(
            name="rnn_dropout",
            widget=UI.Labeled(
                W.Label("RNN dropout: ", layout=label_layout),
                W.FloatText(value=cfg.rnn_dropout, min=0., max=.999, step=.01,
                            layout={"width": "100px", }),
                W.HBox(layout=param_layout)
            )
        )
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                selected_index=0, layout=W.Layout(margin="0 auto 0 0", width="500px"))
    view.set_title(0, "SampleRNN Config")
    return view
