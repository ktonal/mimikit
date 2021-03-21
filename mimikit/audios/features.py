import torchaudio.functional as F
import torchaudio.transforms as T
import torch
#from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
import contextlib
import numpy as np
from abc import ABC

from ..audios import transforms as A
from ..h5data.write import write_feature

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class Feature(ABC):
    @staticmethod
    def extract(path, **kwargs):
        pass

    @staticmethod
    def after_make(db):
        pass

    @staticmethod
    def encode(inputs: torch.Tensor, **kwargs):
        pass

    @staticmethod
    def decode(outputs: torch.Tensor, **kwargs):
        pass


class QuantizedSignal(Feature):

    @staticmethod
    def extract(path, sr=16000, q_levels=255, emphasis=0., sample_encoding='mu_law', normalize=True):
        signal = A.FileTo.signal(path, sr)
        if emphasis:
            signal = A.emphasize(signal, emphasis)
        if sample_encoding == 'mu_law':
            shaper = 'mu_law'
            signal = A.SignalTo.mu_law_compress(signal, q_levels=q_levels, normalize=normalize)
        elif sample_encoding == 'adapted':
            signal, shaper = A.SignalTo.adapted_uniform(signal, q_levels=q_levels,
                                                                             normalize=normalize)
        elif sample_encoding == 'pcm':
            shaper = 'pcm'
            signal = A.SignalTo.pcm_unsigned(signal, q_levels=q_levels, normalize=normalize)
        else:
            raise ValueError("sample_encoding has to 'mu_law', 'adapted', or 'pcm'")
        return dict(qx=(dict(sr=sr,
                             q_levels=q_levels,
                             emphasis=emphasis,
                             shaper=shaper,
                             sample_encoding=sample_encoding),
                        signal.reshape(-1, 1), None))

    # Maybe normalize should be False, since this could be used on small signal snippets
    @staticmethod
    def encode(inputs: torch.Tensor, q_levels=256, emphasis=0., sample_encoding='mu_law', normalize=True, shaper=None):
        if emphasis:
            inputs = F.lfilter(inputs,
                               torch.tensor([1, 0]).to(inputs),  # a0, a1
                               torch.tensor([1, -emphasis]).to(inputs))  # b0, b1
        if normalize:
            inputs = inputs / torch.norm(inputs, p=float("inf"))
        if sample_encoding == 'mu_law':
            return F.mu_law_encoding(inputs, q_levels)
        elif sample_encoding == 'adapted':
            ids = torch.from_numpy(shaper[1]).float()
            # unfortunately Interp1d does not let you select an axis - so process columns one by one
            #signal = torch.stack([shaper_func(inputs[:,k]) for k in range(inputs.shape[1])]).T
            return ((inputs + 1.0) * 0.5 * (q_levels - 1)).astype(np.int)
        elif sample_encoding == 'pcm':
            return ((inputs + 1.0) * 0.5 * (q_levels - 1)).astype(np.int)
        else:
            raise ValueError("sample_encoding has to 'mu_law', 'adapted', or 'pcm'")

    @staticmethod
    def decode(outputs: torch.Tensor, q_levels=256, emphasis=0., sample_encoding='mu_law', shaper=None):
        if sample_encoding == 'mu_law':
            signal = F.mu_law_decoding(outputs, q_levels)
        elif sample_encoding == 'adapted':
            ids = torch.from_numpy(shaper[1]).float().to(outputs)
            xvals = 2 * ids / ids[-1] - 1.0
            # unfortunately Interp1d does not let you select an axis - so process columns one by one
            signal = torch.stack([Interp1d()(xvals, torch.from_numpy(shaper[0]).float().to(outputs), 2.0 * outputs[:,k] / (q_levels - 1) - 1.0) for k in range(outputs.shape[1])]).T
        elif sample_encoding == 'pcm':
            signal = 2.0 * outputs.float() / (q_levels - 1) - 1.0
        else:
            raise ValueError("sample_encoding has to 'mu_law', 'adapted', or 'pcm'")
        if emphasis:
            signal = F.lfilter(signal,
                               torch.tensor([1, -emphasis]).to(signal),  # a0, a1
                               torch.tensor([1 - emphasis, 0]).to(signal))  # b0, b1
        return signal


class MagSpec(Feature):

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        y = A.FileTo.signal(path, sr)
        fft = A.SignalTo.mag_spec(y, n_fft, hop_length)
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        return dict(fft=(params, fft.T, None))

    @staticmethod
    def encode(inputs: torch.Tensor, n_fft=2048, hop_length=512):
        stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.,
                             wkwargs=dict(device=inputs.device))
        return stft(inputs).transpose(-1, -2).contiguous()

    @staticmethod
    def decode(outputs: torch.Tensor, n_fft=2048, hop_length=512):
        gla = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, power=1.,
                           wkwargs=dict(device=outputs.device))
        return gla(outputs.transpose(-1, -2).contiguous())


class SegmentLabels(Feature):

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        feat_dict = MagSpec.extract(path, n_fft, hop_length, sr)
        regions = A.MagSpecTo.regions(feat_dict["fft"][1].T)
        return dict(fft=(dict(n_fft=n_fft, hop_length=hop_length, sr=sr),
                         feat_dict["fft"][1],
                         regions))

    @staticmethod
    def after_make(db):
        labels = np.hstack([np.ones((tp.duration, ), dtype=np.int) * tp.Index
                         for tp in db.fft.regions.itertuples()])
        write_feature(db.h5_file,
                      "labels", dict(n_classes=len(db.fft.regions)), labels)
        files_labels = np.hstack([np.ones((tp.duration, ), dtype=np.int) * tp.Index
                                 for tp in db.fft.files.itertuples()])
        write_feature(db.h5_file,
                      "files_labels", dict(n_classes=len(db.fft.files)), files_labels)
