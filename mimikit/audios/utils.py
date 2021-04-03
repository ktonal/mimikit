import numpy as np
import torch
import librosa


def peakdetector(sig, att, rel):
    """
    Envelope follower

    Parameters
    ----------
    sig : array_like
        the input signal.  For audio signals the abs should be taken or the signal
        should be raised to the power of two
    att : float
        attack coefficient
    rel : float
        release coefficient
    """
    lev = 0.0
    outsig = np.zeros_like(sig)
    for i in range(len(sig)):
        x = sig[i]
        if x > lev:
            lev = lev + att * (x - lev)
        else:
            lev = lev + rel * (x - lev)
        outsig[i] = lev
    return outsig


class EnvFeeder:

    # the prompt env should be one point longer than the prompt data
    def __init__(self, prompt_env, n_steps, att, rel, env_gain=0.1):
        self.att = att
        self.rel = rel
        self.env_gain = env_gain
        self.step = prompt_env.shape[1] - 1
        # the actual env that is fed as input to the network
        self.env = torch.from_numpy(np.concatenate([prompt_env,
                                                    np.zeros((1, n_steps - prompt_env.shape[1], 2))], 1)).float()
        self.lev = torch.from_numpy(prompt_env[:, self.step - 2, 0].copy()).float()

    def next(self, spec, newenv):
        step = self.step
        x = self.env_gain * torch.sum(abs(spec), 1)
        for i in range(len(self.lev)):
            self.lev[i] += self.att * (x[i] - self.lev[i]) if x[i] > self.lev[i] else self.rel * (x[i] - self.lev[i])
        self.env[:, step - 1, 0] = self.lev
        self.env[:, step - 2, 1] = 3 * (self.lev - self.env[:, step - 3, 0])
        step += 1
        x0 = self.env[:, step - 1, 0]
        self.env[:, step + 1, 0] = newenv
        self.env[:, step, 1] = 3 * (newenv - x0)
        self.step = step

    def to(self, device):
        self.env.to(device)
        self.lev.to(device)

# Modified GriffinLim Algorithm that works well with initial approximation.
# code from https://github.com/rbarghou/pygriffinlim/tree/master/pygriffinlim


def modified_fast_griffin_lim_generator(
        spectrogram,
        iterations,
        approximated_signal=None,
        alpha_loc=.1,
        alpha_scale=.4,
        stft_kwargs={},
        istft_kwargs={}):
    """
    :param spectrogram:
    :param iterations:
    :param approximated_signal:
    :param alpha_loc:
    :param alpha_scale:
    :param stft_kwargs:
    :param istft_kwargs:
    :return:
    """
    _M = spectrogram
    for k in range(iterations):
        if approximated_signal is None:
            _P = np.random.randn(*_M.shape)
        else:
            _D = librosa.stft(approximated_signal, **stft_kwargs)
            _P = np.angle(_D)

        _D = _M * np.exp(1j * _P)
        alpha = np.random.normal(alpha_loc, alpha_scale)
        _M = spectrogram + (alpha * np.abs(_D))
        approximated_signal = librosa.istft(_D, **istft_kwargs)
        yield approximated_signal


def mfgla(
        spectrogram,
        iterations,
        approximated_signal=None,
        alpha_loc=.1,
        alpha_scale=.4,
        stft_kwargs={},
        istft_kwargs={}):
    """
    :param spectrogram:
    :param iterations:
    :param approximated_signal:
    :param alpha_loc:
    :param alpha_scale:
    :param stft_kwargs:
    :param istft_kwargs:
    :return:
    """
    generator = modified_fast_griffin_lim_generator(
        spectrogram, iterations, approximated_signal, alpha_loc, alpha_scale, stft_kwargs, istft_kwargs)
    for approximated_signal in generator:
        pass
    return approximated_signal
