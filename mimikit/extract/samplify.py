import dataclasses as dtc
import numpy as np
import librosa
import matplotlib.pyplot as plt

from ..features.functionals import Derivative, Envelop, Interpolate, Functional, Identity

__all__ = [
    "Samplifyer"
]


@dtc.dataclass
class _EnvelopAndGrad:
    """ compute an envelop and its grad"""
    n_fft: int
    overlap: int
    grad_max_lag: int

    def __post_init__(self):
        self.env_ex = Envelop(self.n_fft, self.n_fft//self.overlap,
                              # we need the grad before we interp to time dom
                              normalize=True, interp_to_time_domain=False)
        self.interp = Interpolate(axis=-1, mode="quadratic")
        self.dx = Derivative(self.grad_max_lag, normalize=True)
        self.grad: np.ndarray = None
        self.env: np.ndarray = None
        self.T = 0
        self.y = None

    def fit(self, y):
        self.interp.length = self.T = y.shape[-1]
        self.y = y
        self.env = self.env_ex(y)
        self.grad = self.dx(self.env[None, :])[0]
        self.env, self.grad = self.interp(self.env), self.interp(self.grad)
        return self


class Periods:
    """
    compute indices (attack begin & peak)
    and metrics from a gradient/oscillating signal
    """

    def __init__(self):
        self.y = None
        # attack begin
        self.att_i = None
        # peak
        self.dec_i = None
        # metrics
        self.att_h = None
        self.dec_d = None
        self.pos_sum = None
        self.neg_sum = None
        self.pos_dur = None
        self.neg_dur = None
        self.T = 0

    def fit(self, y):
        self.y = y
        self.T = T = y.shape[-1]
        z_i = ((y[:-1] < 0) & (y[1:] > 0)).nonzero()[0] + 1
        dec_i = np.zeros_like(z_i)
        att_h, dec_d = np.zeros_like(z_i, dtype=float), np.zeros_like(z_i, dtype=float)
        p_sum, n_sum = np.zeros_like(z_i, dtype=float), np.zeros_like(z_i, dtype=float)
        p_dur, n_dur = np.zeros_like(z_i, dtype=int), np.zeros_like(z_i, dtype=int)
        for n, (a, b) in enumerate(zip(z_i, np.r_[z_i, T - 1][1:])):
            att_h[n] = y[a:b].max()
            dec_d[n] = y[a:b].min()
            p_mask = y[a:b] >= 0
            n_mask = y[a:b] < 0
            p_sum[n] = (y[a:b][p_mask]).sum()
            n_sum[n] = -(y[a:b][n_mask]).sum()
            p_dur[n] = p_mask.sum()
            n_dur[n] = n_mask.sum()
            try:
                dec_i[n] = ((y[a:b - 1] > 0) & (y[a + 1:b] < 0)).nonzero()[0][0] + a
            except:
                dec_i[n] = a + 1
        self.att_i = z_i
        self.dec_i = dec_i
        self.att_h = att_h
        self.dec_d = dec_d
        self.pos_sum = p_sum
        self.neg_sum = n_sum
        self.pos_dur = p_dur
        self.neg_dur = n_dur
        return self


@dtc.dataclass
class Samplifyer(Functional):
    filter_level: int = 0
    sensitivity: float = .5

    def __post_init__(self):
        self.y = None
        self.T = None
        if self.filter_level > 4 or self.filter_level < 0:
            raise ValueError("filter_level must be between 0 and 4")

        self.levels = \
            [
                _EnvelopAndGrad(n_fft=8192, overlap=32, grad_max_lag=9),
                _EnvelopAndGrad(n_fft=4096, overlap=64, grad_max_lag=33),
                _EnvelopAndGrad(n_fft=2048, overlap=32, grad_max_lag=17),
                _EnvelopAndGrad(n_fft=1024, overlap=16, grad_max_lag=9),
                _EnvelopAndGrad(n_fft=512, overlap=8, grad_max_lag=9),
                _EnvelopAndGrad(n_fft=256, overlap=8, grad_max_lag=9)
            ][self.filter_level:]
        self.coarse_env, self.coarse_grad = None, None
        self.coarse_cuts = None
        self.scores = None
        self.cuts = None
        self.sides = None
        self.fine_envs = None
        self.windows = None

    def np_func(self, y):
        return self.label(y)

    def label(self, y):
        cuts = self.fit(y).cuts
        labels = np.zeros_like(y, dtype=int)
        labels[cuts] = 1
        return np.cumsum(labels)

    def fit(self, y):
        self.y = y
        self.T = y.shape[0]

        # I. build the different envelops
        for d in self.levels:
            d.fit(y)
        coarse_level = self.levels[0]
        self.coarse_env = coarse_level.env
        self.coarse_grad = coarse_level.grad

        # II. filter attacks at the coarse level:
        # ratio (coarse_env[peak] - coarse_env[attack_begin]) / coarse_env[attack_begin]
        # must be ABOVE self.sensitivity
        per = Periods().fit(self.coarse_grad)
        scores = ((self.coarse_env[per.dec_i] - self.coarse_env[per.att_i])
                  / self.coarse_env[per.att_i])
        bool_mask = scores > self.sensitivity
        # TODO: enforce `min_length` : merge segment too small to the smallest of {left, right}
        # TODO? min_start, max_start, min_peak, max_peak, min_incr, max_incr...?
        self.scores = scores[bool_mask]
        self.coarse_cuts = per.att_i[bool_mask]
        self.coarse_peaks = per.dec_i[bool_mask]
        self.mins_rank = np.zeros_like(self.coarse_cuts)
        self.maxs_rank = np.zeros_like(self.coarse_cuts)
        self.mins_rank[self.coarse_env[self.coarse_cuts].argsort()] = np.arange(len(self.coarse_cuts))
        self.maxs_rank[(1 - self.coarse_env[self.coarse_peaks]).argsort()] = np.arange(len(self.coarse_peaks))

        # III. refine he cuts
        fine_grads = [level.grad for level in self.levels[1:]]
        self.fine_envs = fine_envs = [level.env for level in self.levels[1:]]

        # a) TODO: find out if we need to refine left or right of the coarse cuts
        # (we pick the side that maximizes coarse_env - finer_envs)
        self.windows = half_window = np.minimum(self.coarse_peaks - self.coarse_cuts, 2000)
        left_scores = np.zeros_like(self.coarse_cuts, dtype=float)
        right_scores = np.zeros_like(self.coarse_cuts, dtype=float)
        for i, (c, w) in enumerate(zip(self.coarse_cuts, half_window)):
            left = slice(c - w, c)
            right = slice(c, c + w)
            for env in fine_envs[-1:]:
                left_scores[i] += (self.coarse_env[left] - env[left]).max()
                right_scores[i] += (self.coarse_env[right] - env[right]).max()
        self.sides = sides = np.stack((left_scores, right_scores)).argmax(axis=0)

        # b) refine
        self.cuts = np.zeros_like(self.coarse_cuts)
        z_crossings = librosa.zero_crossings(y)

        for i in range(len(self.coarse_cuts)):
            c = self.coarse_cuts[i]
            d = self.coarse_peaks[i]
            if sides[i] == 0:
                d = c
                c = c - (d - c)
            # refine
            for env, grad in zip(fine_envs, fine_grads):
                c, d = self._refine(c, d, env, grad)

            # snap to nearest zero crossing
            before, after = c, c + 1
            while not z_crossings[before] and not z_crossings[after]:
                before -= 1
                after += 1

            self.cuts[i] = before if z_crossings[before] else after

        return self

    @staticmethod
    def _refine(start, stop, env, grad):
        if start == stop:
            return start, stop
        e = env[start:stop]
        g = grad[start:stop]
        new_start = (.9 * e + .1 * (1 - g)).argmin()
        new_start = int(new_start < stop - 1) * new_start
        return new_start + start, max(e.argmax() + start, new_start + start)

    def plot_refined_cuts(self):
        for c, c_hat, w, side, score in zip(self.coarse_cuts, self.cuts, self.windows, self.sides, self.scores):
            left = min(c - w, c_hat - w)
            right = max(c + w, c_hat + w)
            t = slice(left, right)
            plt.figure()
            plt.plot(self.y[t], label='signal', alpha=.5)
            plt.plot(self.coarse_env[t], label=f'level {self.filter_level}')
            for i, env in enumerate(self.fine_envs, 1):
                plt.plot(env[t], label=f"level {self.filter_level + i}")
            plt.scatter(c - left, 0,
                        marker='X', color='red', s=200, label=f"raw cut (score={score:.2f})")
            plt.scatter(c_hat - left, 0,
                        marker='X', color='green', s=200, label=f"fine cut ({['left', 'right'][side]})")
            plt.legend(loc='upper left')

    def export_with_silence(self, insert_sec=1., sr=44100):
        return np.concatenate(
            [np.r_[x, np.zeros(int(sr * insert_sec))] for x in self.export_as_list()]
        )

    def export_as_list(self):
        return np.split(self.y, self.cuts)

    def torch_func(self, inputs):
        raise NotImplementedError

    def inv(self) -> "Functional":
        return Identity()


def samplify(audio):
    sr, y = audio
    y = librosa.util.normalize(y.astype(np.float32), )
    samplifyer = Samplifyer(filter_level=1, sensitivity=.1, sr=44100)
    samplifyer.fit(y)

    fig = plt.figure(figsize=(50, 6))

    plt.plot(y, alpha=.666, label='signal')

    plt.plot(samplifyer.coarse_env / samplifyer.coarse_env.max(),
             label='coarse envelop', c='violet', alpha=.85)

    plt.show()
