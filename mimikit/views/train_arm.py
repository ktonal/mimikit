import ipywidgets as W
from .. import ui as UI
from ..loops.train_loops import TrainARMConfig

__all__ = [
    "train_arm_view"
]


def train_arm_view(cfg: TrainARMConfig):
    view = UI.ConfigView(
        cfg,
        UI.Param(name='root_dir',
                 widget=UI.Labeled(
                     "Directory",
                     W.Text(value=cfg.root_dir)
                 ),
                 position=(0, 0)
                 ),
        UI.Param(name='_',
                 widget=W.HTML("<h4 style='text-align: end'>Batches</h4>", layout=dict(height='28px')),
                 position=(1, 0)),
        UI.Param(name='_',
                 widget=W.HTML("<hr>", layout=dict(height='28px')),
                 position=(1, 1)),
        UI.Param(name="batch_size",
                 widget=UI.Labeled(
                     "Batch Size: ",
                     UI.pw2_widget(cfg.batch_size),
                 ),
                 setter=lambda conf, v: int(v),
                 position=(2, 0)
                 ),
        UI.Param(name="batch_length",
                 widget=UI.Labeled(
                     "Batch Length: ",
                     UI.pw2_widget(cfg.batch_length),
                 ),
                 setter=lambda conf, v: int(v),
                 position=(2, 1),
                 ),
        UI.Param(name="downsampling",
                 widget=UI.Labeled(
                     "Batches downsampling",
                     W.IntText(value=cfg.downsampling)
                 ),
                 position=(3, 0)),
        UI.Param(name="oversampling",
                 widget=UI.Labeled(
                     "Batch oversampling",
                     W.IntText(value=cfg.oversampling)
                 ),
                 position=(3, 1)),
        UI.Param(name='tbptt_chunk_length',
                 widget=UI.Labeled(
                     "TBPTT length",
                     W.IntText(value=cfg.tbptt_chunk_length)
                 ),
                 position=(4, 0)),
        UI.Param(name='_',
                 widget=W.HTML("<h4 style='text-align: end'>Epochs</h4>", layout=dict(height='28px')),
                 position=(5, 0)),
        UI.Param(name='_',
                 widget=W.HTML("<hr>", layout=dict(height='28px')),
                 position=(5, 1)),
        UI.Param(name="max_epochs",
                 widget=UI.Labeled(
                     "Number of Epochs: ",
                     W.IntText(value=cfg.max_epochs),
                     "training will be performed for this number of epochs."
                 ),
                 position=(6, 0)),
        UI.Param(name="limit_train_batches",
                 widget=UI.Labeled(
                     "Max batches per epoch: ",
                     W.IntText(value=0 if cfg.limit_train_batches is None else cfg.limit_train_batches, ),
                     "limit the number of batches per epoch, enter 0 for no limit"
                 ),
                 setter=lambda conf, ev: ev if ev > 0 else None,
                 position=(6,1)),
        UI.Param(name='_',
                 widget=W.HTML("<h4 style='text-align: end'>Optimizer</h4>", layout=dict(height='28px')),
                 position=(7, 0)),
        UI.Param(name='_',
                 widget=W.HTML("<hr>", layout=dict(height='28px')),
                 position=(7, 1)),
        UI.Param(name="max_lr",
                 widget=UI.Labeled(
                     "Learning Rate: ",
                     W.FloatSlider(
                         value=cfg.max_lr, min=1e-5, max=1e-2, step=.00001,
                         readout_format=".2e",
                     ),
                 ),
                 position=(8, 0)
                 ),
        UI.Param(name="betas",
                 widget=UI.Labeled(
                     "Beta 1",
                     W.FloatLogSlider(
                         value=cfg.betas[0], min=-.75, max=0., step=.001, base=2,
                     ),
                 ),
                 setter=lambda conf, ev: (ev, conf.betas[1]),
                 position=(9, 0)
                 ),
        UI.Param(name="betas",
                 widget=UI.Labeled(
                     "Beta 2",
                     W.FloatLogSlider(
                         value=cfg.betas[1], min=-.75, max=0., step=.001, base=2,),
                 ),
                 setter=lambda conf, ev: (conf.betas[0], ev),
                 position=(9, 1)
                 ),
        UI.Param(name='_',
                 widget=W.HTML("<h4 style='text-align: end'>LR Scheduler</h4>",
                               layout=dict(height='28px')),
                 position=(10, 0)),
        UI.Param(name='_',
                 widget=W.HTML("<hr>", layout=dict(height='28px')),
                 position=(10, 1)),
        UI.Param(name="div_factor",
                 widget=UI.Labeled(
                     "Start LR div factor",
                     W.FloatSlider(
                         value=cfg.div_factor, min=0.001, max=100., step=.001,
                     ),
                 ),
                 position=(11, 0)
                 ),
        UI.Param(name="final_div_factor",
                 widget=UI.Labeled(
                     "End LR div factor",
                     W.FloatSlider(
                         value=cfg.final_div_factor, min=0.001, max=100., step=.001,
                     ),
                 ),
                 position=(11, 1)
                 ),
        UI.Param(name="pct_start",
                 widget=UI.Labeled(
                     "Percent training start LR to max LR",
                     W.FloatSlider(
                         value=cfg.pct_start, min=0.0, max=1., step=.001,
                     ),
                 ),
                 position=(12, 0)
                 ),
        UI.Param(name='_',
                 widget=W.HTML("<h4 style='text-align: end'>Tests & Checkpoints</h4>", layout=dict(height='28px')),
                 position=(13, 0)),
        UI.Param(name='_',
                 widget=W.HTML("<hr>", layout=dict(height='28px')),
                 position=(13, 1)),
        UI.Param(name="every_n_epochs",
                 widget=UI.Labeled(
                     "Test/Checkpoint every $N$ epochs",
                     W.IntText(value=cfg.every_n_epochs),
                 ),
                 position=(14, 0)),
        UI.Param(name='n_examples',
                 widget=UI.Labeled(
                     "$N$ Test examples",
                     W.IntText(value=cfg.n_examples),
                 ),
                 position=(14, 1)),
        UI.Param(name='prompt_length_sec',
                 widget=UI.Labeled(
                     "Prompt length (in sec.)",
                     W.FloatText(value=cfg.prompt_length_sec),
                 ),
                 position=(15, 0)),
        UI.Param(name='outputs_duration_sec',
                 widget=UI.Labeled(
                     "Tests length (in sec.)",
                     W.FloatText(value=cfg.outputs_duration_sec),
                 ),
                 position=(15, 1)),
        UI.Param(name='temperature',
                 widget=UI.Labeled(
                     "Test examples' temperatures",
                     W.Text(value='' if cfg.temperature is None else str(cfg.temperature)[1:-1])
                 ),
                 position=(16, slice(0, 2)),
                 setter=lambda config, ev: tuple(map(eval, ev.split(', '))) if ev else None),
        UI.Param(name="CHECKPOINT_TRAINING",
                 widget=UI.Labeled(
                     "Checkpoint Training: ",
                     UI.yesno_widget(initial_value=cfg.CHECKPOINT_TRAINING),
                 ),
                 position=(17, 0)),
        UI.Param(name="MONITOR_TRAINING",
                 widget=UI.Labeled(
                     "Monitor Training: ",
                     UI.yesno_widget(initial_value=cfg.MONITOR_TRAINING),
                 ),
                 position=(17, 1)),
        grid_spec=(18, 2)
    ).as_widget(lambda children, **kwargs: W.Accordion([W.VBox(children=children)], **kwargs),
                selected_index=0, layout=W.Layout(margin="0 auto 0 0", width="100%"))
    view.set_title(0, "Optimization Loop")
    return view
