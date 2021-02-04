import librosa
import soundfile as sf
import torch
import numpy as np
from .pbind import Event


# Maybe it would be better to set the hop_size by some other means.
# This is prone to errors if you don't set the hop_size to the size
# wich which the data for a given model was prepared
Event.default_parent = {'n_frames': lambda ev: round(ev['dur'] * ev['srate'] / ev['hop_size']),
                        'start_frame': lambda ev: round(ev['start'] * ev['srate'] / ev['hop_size']),
                        'srate': 22050,
                        'blend_nbr': 0.0,
                        'hop_size': 512
                        }

# Further ideas to implement:
# it would be interesting to be able to switch models based on some
# kind of onset detection
#
# event type 'choke' which tries by some way to push the model into decay
# and go into silence - this cannot work with all models but might work 
# on some.

class AREnsembleGenerator:
    """
    Generation of audio data from multiple trained auto-regressive models.
    The class uses Pbind patterns to control the model selection and generation parameters.
    """

    def __init__(self, models, prompt_data,
                 max_time, pattern,
                 device='cpu', tensor_api='torch', prepad=16):
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
        self.num_bins = prompt_data.shape[-1]
        if tensor_api == 'torch':
            self.noise_fn = lambda x: torch.randn(x).to(self.device)
            self.cat_fn = lambda x: torch.cat(x, 1)
            self.convert_fn = lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            self.zero_fn = lambda x: torch.zeros(x).to(self.device)
            # Prepend zeros to ensure there is enough data independent
            # of the prompt size.  This is not ideal, but avoids having to
            # check whether enough data is available at every generation step.
            self.output = torch.zeros((1, prepad, prompt_data.shape[-1]), dtype=torch.float32).to(self.device)
        elif tensor_api == 'numpy':
            self.noise_fn = np.randn
            self.zero_fn = np.zeros
            self.cat_fn = lambda x: np.concatenate(x, axis=1)
            self.convert_fn = lambda x: x
            self.output = np.zeros((1, prepad, prompt_data.shape[-1]), dtype=np.float32)
        else:
            print('tensor_api has to be numpy or torch')
        for m in self.models:
            m.eval()
            m.to(self.device)

    # def multi_fftsize_generate(self, hop_size=512, start_event=Event({})):
    #     out_sig = np.array([], dtype=np.float32)
    #     stream = self.pattern.asStream()
    #     n_bins = 0
    #     while self.time < self.max_time:
    #         event = stream.next(start_event)
    #         if event['n_bins'] != n_bins:
    #             n_bins = event['n_bins']
    #         event_type = event['type']
    #         getattr(self, event_type)(event)
    #     out_sig = np.concatenate([out_sig, librosa.griffinlim(self.output.cpu().numpy()[0].T, 64, hop_size)])
    #     return res

    def generate(self, time_domain=True, hop_size=512, start_event=Event({})):
        stream = self.pattern.asStream()
        while self.time < self.max_time:
            event = stream.next(start_event)
            event_type = event['type']
            getattr(self, event_type)(event)
        res = librosa.griffinlim(self.output.cpu().numpy()[0].T, 64, hop_size)
        return res

    def insert(self, event):
        n_frames = event.value('n_frames')
        data = event['data']
        self.output = self.cat_fn([self.output, self.convert_fn(data)])
        self.time += n_frames * event['hop_size'] / event['srate']

    def rest(self, event):
        n_frames = event.value('n_frames')
        frames = self.zeros_fn((1, n_frames, self.num_bins))
        self.output = self.cat_fn([self.output, self.convert_fn(frames)])
        self.time += n_frames * event['hop_size'] / event['srate']

    def prompt(self, event):
        n_frames = event.value('n_frames')
        start_frame = event.value('start_frame')
        data = event['data']
        if data is None:
            data = self.prompt_data
        data = data[start_frame:start_frame + n_frames]
        self.output = self.cat_fn([self.output, self.convert_fn(data)])
        self.time += n_frames * event['hop_size'] / event['srate']

    def original(self, event):
        # insert a piece of the orginal data into the output
        # if nbrs are used select the starting point by finding
        # nearest neighbor
        n_frames = event.value('n_frames')
        nbrs = event['nbrs']
        data = event['data']
        if data is None:
            data = self.prompt_data
        if nbrs:
            cur_frame = self.output[0, -1:]
            _, inds = nbrs.kneighbors(cur_frame)
            inds = inds[0, 0] + 1
            inds = np.clip(inds, 0, data.shape[0] - n_frames)
            data = data[inds:inds + n_frames]
            self.output = self.cat_fn([self.output, self.convert_fn(data)])
        else:
            print("event type original needs to have nbrs.")
        self.time += n_frames * event['hop_size'] / event['srate']

    def model(self, event):
        n_frames = event.value('n_frames')
        model_num = event['model']
        decay = event['decay'] or 1.0
        noise = event['noise']
        interpolate = event['interpolate'] or 1
        repeat = event['repeat']
        blend_nbr = event['blend_nbr']

        # in case we have multiple models (a list or tuple in model_num)  we blend them
        # the noise parameter is currently ignored when blending
        if hasattr(model_num, "__len__"):
            # maybe not the best solution: just set blend to make
            # a compromise between averaging and just adding.
            # there is no specific reason for the choice of [0.9, 0.9]
            blend = event['blend'] or [0.9, 0.9]
            frame_num = 0
            # dists, inds = self.nbrs.kneighbors(x)
            with torch.no_grad():
                while frame_num < n_frames:
                    model = self.models[model_num[0] % len(self.models)]
                    in_slice, out_slice = model.generation_slices()
                    next_input = self.output[:, in_slice]
                    next_frame = model(next_input)[:, out_slice] * blend[0]
                    for bi, m in enumerate(model_num[1:]):
                        model = self.models[m % len(self.models)]
                        in_slice, out_slice = model.generation_slices()
                        next_input = self.output[:, in_slice]
                        next_frame += model(next_input)[:, out_slice] * blend[bi + 1]
                    next_frame = next_frame * decay
                    if blend_nbr > 0.0:
                        nbrs = event['nbrs']
                        data = event['data']
                        _, inds = nbrs.kneighbors(next_frame[0])
                        inds = inds[0, 0]
                        next_frame += blend_nbr * (self.convert_fn(data[inds:inds+1]) - next_frame)
                    if interpolate > 1:
                        cur_frame = self.output[:, -1:]
                        left = round(n_frames - frame_num)
                        n = min(left, interpolate)
                        for k in range(n):
                            frac = (k + 1) / interpolate
                            frame = frac * next_frame + (1.0 - frac) * cur_frame
                            self.output = self.cat_fn([self.output, frame])
                        frame_num += n
                    elif repeat:
                        left = round(n_frames - frame_num)
                        n = min(left, interpolate)
                        for k in range(n):
                            self.output = self.cat_fn([self.output, next_frame])
                        frame_num += n
                    else:
                        self.output = self.cat_fn([self.output, next_frame])
                        frame_num += 1
                self.time += n_frames * event['hop_size'] / event['srate']
                print(n_frames, self.time)
            return self

        ################################################################################
        # code only reached when not blending models
        ################################################################################

        model = self.models[model_num % len(self.models)]
        in_slice, out_slice = model.generation_slices()

        if (interpolate > 1) or repeat:
            frame_num = 0
            with torch.no_grad():
                while frame_num < n_frames:
                    next_input = self.output[:, in_slice]
                    if noise:
                        noise_sig = self.noise_fn(next_input.shape) * noise
                    else:
                        noise_sig = 0.0
                    next_frame = model(next_input + noise_sig)[:, out_slice] * decay
                    if blend_nbr > 0.0:
                        nbrs = event['nbrs']
                        data = event['data']
                        _, inds = nbrs.kneighbors(next_frame[0])
                        inds = inds[0, 0]
                        next_frame += blend_nbr * (self.convert_fn(data[inds:inds+1]) - next_frame)
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
                    if blend_nbr > 0.0:
                        nbrs = event['nbrs']
                        data = event['data']
                        _, inds = nbrs.kneighbors(next_frame[0])
                        inds = inds[0, 0]
                        next_frame += blend_nbr * (self.convert_fn(data[inds:inds+1]) - next_frame)
                    self.output = self.cat_fn([self.output, next_frame])
        else:
            with torch.no_grad():
                for n in range(n_frames):
                    next_input = self.output[:, in_slice]
                    next_frame = model(next_input)[:, out_slice] * decay
                    if blend_nbr > 0.0:
                        nbrs = event['nbrs']
                        data = event['data']
                        _, inds = nbrs.kneighbors(next_frame[0])
                        inds = inds[0, 0]
                        next_frame += blend_nbr * (self.convert_fn(data[inds:inds+1]) - next_frame)
                    self.output = self.cat_fn([self.output, next_frame])

        self.time += n_frames * event['hop_size'] / event['srate']
        print(n_frames, self.time)

