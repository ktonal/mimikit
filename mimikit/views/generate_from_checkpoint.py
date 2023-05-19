from peaksjs_widget import PeaksJSWidget, Segment
from ipywidgets import widgets as W
import IPython.display as ipd

from .. import ui as UI
from .timestamps_selector import TimeStampsSelector
from ..loops.generate import GenerateLoopV2
from ..features.dataset import DatasetConfig
from ..config import Config
from ..checkpoint import Checkpoint

__all__ = [
    "GenerateFromCheckpointView"
]


# TODO: Reset prompts, destroy outputs, fix only one file selected class,
#  reload everything after ckpt selection, time in prompt label
# noinspection PyTypeChecker
class GenerateFromCheckpointView:

    def __init__(self, root_dir="./"):
        self.title = W.HTML("<h4>Select Checkpoint</h4>", layout=dict(margin='0 0 0 8px'))
        self.picker = UI.CheckpointPicker(root=root_dir, multiple=False,
                                          grid_height="100px")
        self.load_ckpt = W.Button(description="Load")
        self.ckpt = None
        self.prompt_selector: TimeStampsSelector = None
        self.sr = None
        self.output_area = W.Output()
        self.n_outputs = 0

        self.prompt_length_w = W.FloatText(value=1., step=.01)
        self.outputs_length_w = W.FloatText(value=30., step=.01)
        self.batch_size_w = W.IntText(value=8)
        self.downsampling = 1
        self.generate_w = W.Button(description="Generate")

        self.load_ckpt.on_click(self.load_callback)
        self.container = W.VBox(children=(self.title,
                                          self.picker.widget,
                                          self.load_ckpt,
                                          ))

    @property
    def widget(self):
        return self.container

    def load_callback(self, ev):
        path = self.picker.selected
        if path:
            self.ckpt = Checkpoint.from_path(path)
            extractor = Config.deserialize(self.ckpt.dataset.attrs["config"], as_type=DatasetConfig).extractors[0]
            self.sr = extractor.functional.unit.sr
            self.downsampling = self.ckpt.training_config.training.downsampling
            name = extractor.name
            self.prompt_selector = TimeStampsSelector(array=getattr(self.ckpt.dataset, name)[:], sr=self.sr)
            title = W.HTML(
                """<h4>Select Prompts</h4>""",
                layout=dict(margin='0 0 0 8px'))
            self.generate_w.on_click(self.generate_callback)
            widgets = W.VBox(children=(
                UI.Labeled("prompt lengt (sec.)", self.prompt_length_w),
                UI.Labeled("output length (sec.)", self.outputs_length_w),
                UI.Labeled("batch size", self.batch_size_w),
                self.generate_w,
                self.output_area
            ))
            self.container.children = (
                *self.container.children[:3],
                title,
                self.prompt_selector.widget,
                widgets
            )

    def generate_callback(self, ev):
        self.output_area.clear_output()
        if not any(self.prompt_selector.timestamps):
            with self.output_area:
                raise ValueError("Please select some prompt positions before clicking 'Generate'")
            return
        loop_cfg = GenerateLoopV2.Config(
            output_duration_sec=self.outputs_length_w.value,
            prompts_length_sec=self.prompt_length_w.value,
            prompts_position_sec=tuple(t - self.prompt_length_w.value
                                       for t in self.prompt_selector.timestamps),
            batch_size=self.batch_size_w.value,
            downsampling=self.downsampling
            ,
            display_waveform=False,
            yield_inversed_outputs=True
        )
        loop = GenerateLoopV2.from_config(
            loop_cfg,
            self.ckpt.dataset,
            self.ckpt.network
        )
        for output in loop.run():
            for out in output[0]:
                element_id = f"{self.ckpt.id}_ep={self.ckpt.epoch}_output_{self.n_outputs}"
                ipd.display(
                    W.HTML(f"<h6>{element_id}</h6>"),
                    PeaksJSWidget(array=out.cpu().numpy(), sr=self.sr, segments=[
                        Segment(
                            startTime=0.,
                            endTime=self.prompt_length_w.value,
                            id=0,
                            editable=False,
                            labelText="Prompt",
                        )
                    ], with_save_button=True, with_play_button=True, element_id=element_id))
                self.n_outputs += 1
