import ipywidgets as W
from .. import ui as UI
from ..loops.train_loops import TrainARMConfig

__all__ = [
    "train_arm_view"
]


def train_arm_view(cfg: TrainARMConfig):
    return UI.ConfigView(
        cfg,
        UI.Param(name="batch_size",
                 widget=UI.pw2_widget(
                     W.Label(value="Batch Size: ", layout=W.Layout(min_width="max-content", margin="0 0 0 auto"),
                             tooltip="test tooltip"),
                     W.Text(value="64", layout=W.Layout(width="25%"), disabled=False),
                     W.Button(icon="plus", layout=W.Layout(width="25%")),
                     W.Button(icon="minus", layout=W.Layout(width="25%")),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 ),
                 compute=lambda conf, v: int(v)
                 ),
        UI.Param(name="batch_length",
                 widget=UI.pw2_widget(
                     W.Label(value="Batch Length: ", layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     W.Text(value="64", layout=W.Layout(width="25%")),
                     W.Button(icon="plus", layout=W.Layout(width="25%")),
                     W.Button(icon="minus", layout=W.Layout(width="25%")),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 ),
                 compute=lambda conf, v: int(v)
                 ),
        UI.Param(name="max_epochs",
                 widget=UI.Labeled(
                     UI.with_tooltip(W.Label(value="Number of Epochs: ",
                                             layout=W.Layout(min_width="max-content",
                                                             margin="auto 0 auto auto")),
                                     "training will be performed for this number of epochs."),
                     W.IntText(value=cfg.max_epochs,
                               layout={"width": "70%", }),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 )),
        UI.Param(name="max_lr",
                 widget=UI.Labeled(
                     W.Label(value="Learning Rate: ",
                             layout=W.Layout(min_width="max-content", margin="0 0 0 auto")
                             ),
                     W.FloatSlider(
                         value=1e-3, min=1e-5, max=1e-2, step=.00001,
                         readout_format=".2e",
                         layout={"width": "75%"}
                     ),
                     W.HBox(
                         layout=W.Layout(width="100%", margin="8px 0 8px 0")
                     )
                 )),
        UI.Param(name="betas",
                 widget=UI.Labeled(
                     W.Label(value="Beta 1", layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     W.FloatLogSlider(
                         value=.9, min=-.75, max=0., step=.001, base=2,
                         layout={"width": "75%"}),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 ),
                 compute=lambda conf, ev: (ev, conf.betas[1])),
        UI.Param(name="betas",
                 widget=UI.Labeled(
                     W.Label(value="Beta 2", layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     W.FloatLogSlider(
                         value=.9, min=-.75, max=0., step=.001, base=2,
                         layout={"width": "75%"}),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 ),
                 compute=lambda conf, ev: (conf.betas[0], ev)),
        UI.Param(name="CHECKPOINT_TRAINING",
                 widget=UI.yesno_widget(
                     W.Label(value="Checkpoint Training: ",
                             layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     container=W.HBox(layout=W.Layout(width="100%")),
                     initial_value=True,
                     buttons_layout=W.Layout(width="27.5%")
                 )),
        UI.Param(name="MONITOR_TRAINING",
                 widget=UI.yesno_widget(
                     W.Label(value="Monitor Training: ", layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     container=W.HBox(layout=W.Layout(width="100%")),
                     initial_value=True,
                     buttons_layout=W.Layout(width="27.5%")
                 )),
        UI.Param(name="every_n_epochs",
                 widget=UI.Labeled(
                     W.Label(value="Test/Checkpoint every $N$ epochs",
                             layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     W.IntText(value=cfg.every_n_epochs,
                               layout={"width": "100px", }),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 )),
        UI.Param(name='n_examples',
                 widget=UI.Labeled(
                     W.Label("$N$ Test examples", layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     W.IntText(value=cfg.n_examples,
                               layout={"width": "100px", }),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 )),
        UI.Param(name='n_steps',
                 widget=UI.Labeled(
                     W.Label("Tests length (in sec.)", layout=W.Layout(min_width="max-content", margin="0 0 0 auto")),
                     W.IntText(value=16.,
                               layout={"width": "100px", }),
                     W.HBox(layout=W.Layout(width="100%", margin="8px 0 8px 0"))
                 ))
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                titles=("Optimization Loop",), selected_index=0, layout=W.Layout(margin="0 auto 0 0", width="500px"))
