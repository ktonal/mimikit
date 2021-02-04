import librosa
import soundfile as sf
import torch
import numpy as np
from pbind import Event


Event.default_parent = {'n_frames': lambda ev: ev['dur'] * ev['srate'] / ev['hop_size'],
                        'srate': 22050,
                        'hop_size': 512
                        }


class AREnsembleGenerator:
    """
    Class for generating audio data from multiple trained auto-regressive models.
    The class uses Pbind patterns to control the model selection and generation parameters.
    """

    def __init__(self, models, prompt_data, max_time, pattern, device='cpu', tensor_api='torch', prepad=16):
        """
        Parameters
        ----------
        models : list
            List of trained models
        prompt_data : ndarray
            Default prompt data (other prompt data can be used if given in a prompt event)
        max_time : number
            Maximal seconds for generated output.  This value does not specify the exact length of
            the output since with any generation event durations are converted into an integer
            number of frames.
        pattern : Pbind
            A Pbind pattern that controls the generation parameters.
        """
        self.models = models
        self.pattern = pattern
        self.device = device
        self.time = 0.0
        self.prompt_data = prompt_data
        self.max_time = max_time
        if tensor_api == 'torch':
            self.noise_fn = torch.randn
            self.cat_fn = lambda x: torch.cat(x, 1)
            self.convert_fn = lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            # Prepend zeros to ensure there is enough data independent
            # of the prompt size.  This is not ideal, but avoids having to
            # check whether enough data is available at every generation step.
            self.output = torch.zeros((1, prepad, prompt_data.shape[-1]), dtype=torch.float32).to(device)
        elif tensor_api == 'numpy':
            self.noise_fn = np.randn
            self.cat_fn = lambda x: np.concatenate(x, axis=1)
            self.convert_fn = lambda x: x
            self.output = np.zeros((1, prepad, prompt_data.shape[-1]), dtype=np.float32)
        else:
            print('tensor_api has to be numpy or torch')
        for m in self.models:
            m.eval()
            m.to(device)

    def generate(self, start_event = Event({})):
        stream = self.pattern.asStream()
        while self.time < self.max_time:
            event = stream.next(start_event)
            event_type = event['type']
            getattr(self, event_type)(event)
        return self.output

    def append_frames(self, event):
        frames = event['frames']
        self.output = self.cat_fn([self.output, self.convert_fn(frames)])
        self.time += n_frames * event['hop_size'] / event['srate']

    def prompt(self, event):
        n_frames = event.value('n_frames')
        start_frame = event.value('start')
        data = event['data'][start_frame:start_frame + n_frames]
        self.output = self.cat_fn([self.output, self.convert_fn(data)])
        self.time += n_frames * event['hop_size'] / event['srate']

    def model(self, event):
        n_frames = event.value('n_frames')
        model_num = event['model']
        decay = event['decay'] or 1.0
        noise = event['noise']
        interpolate = event['interpolate']
        repeat = event['repeat']
        model = self.models[model_num % len(self.models)]
        in_slice, out_slice = model.generation_slices()
        print('model generate', n_frames, in_slice)

        if interpolate or repeat:
            frame_num = 0
            with torch.no_grad():
                while frame_num < n_frames:
                    next_input = self.output[:, in_slice]
                    if noise:
                        noise_sig = self.noise_fn(next_input.shape) * noise
                    else:
                        noise_sig = 0.0
                    next_frame = model(next_input + noise_sig)[:, out_slice] * decay
                    if interpolate:
                        cur_frame = self.output[:, -1:]
                        left = round(n_frames - frame_num)
                        n = min(left, interpolate)
                        for k in range(n):
                            frac = (k + 1) / interpolate
                            frame = frac * next_frame + (1.0 - frac) * cur_frame
                            self.output = self.cat_fn([self.output, frame])
                        frame_num += n
                    else:
                        left = round(n_frames - frame_num)
                        n = min(left, repeat)
                        for k in range(n):
                            self.output = self.cat_fn([self.output, next_frame])
                        frame_num += n
        elif noise:
            with torch.no_grad():
                for n in range(n_frames):
                    next_input = self.output[:, in_slice]
                    noise_sig = self.noise_fn(next_input.shape) * noise
                    next_frame = model(next_input + noise_sig)[:, out_slice] * decay
                    self.output = self.cat_fn([self.output, next_frame])
        else:
            with torch.no_grad():
                for n in range(n_frames):
                    next_input = self.output[:, in_slice]
                    next_frame = model(next_input)[:, out_slice] * decay
                    self.output = self.cat_fn([self.output, next_frame])

        self.time += n_frames * event['hop_size'] / event['srate']
        print(n_frames, self.time)

