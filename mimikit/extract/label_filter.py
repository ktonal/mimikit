import numpy as np
import dataclasses as dtc
from ..features.functionals import Functional

__all__ = [
    "label_filter",
    "LabelFilter"
]


def _get_counts(labels, R):
    K = R + 1
    if K % 2 == 0:
        K += 1
    l2d = np.lib.stride_tricks.sliding_window_view(np.pad(labels, (K//2, K//2), constant_values=-1), (K,))
    where_l, where_r = np.ones(l2d.shape[0], dtype=bool), np.ones(l2d.shape[0], dtype=bool)
    counts = np.ones(l2d.shape[0], dtype=int)
    center = K//2
    for r in range(1, K//2 + 1):
        where_l[where_l] = (l2d[where_l, center] == l2d[where_l, center - r])
        where_r[where_r] = (l2d[where_r, center] == l2d[where_r, center + r])
        counts[where_l] += 1
        counts[where_r] += 1
    flagged = counts < R
    c2d = np.lib.stride_tricks.sliding_window_view(np.pad(counts, (K//2, K//2), constant_values=0), (K,))
    return l2d, c2d, flagged


def _filter_window(w, elem_counts, glob_counts, label_undecidable):
    e_i = w.shape[0]//2
    elem = w[e_i]
    elem_count = elem_counts[e_i]
    glob_elem_count = glob_counts[elem]
    c_max_i = elem_counts.argmax()
    c_max = elem_counts[c_max_i]
    if c_max == 1:
        # w = np.sort(w)
        w_hat = w[elem_counts == 1]
        gc = glob_counts[w_hat]
        gc_max_i = gc.argmax()
        gc_max = gc[gc_max_i]
        if gc_max == 1 or (gc == gc_max).all():
            # all labels are global singletons
            if label_undecidable:
                v = -1
            elif w[e_i-1] == w[e_i+1]:
                # elem is surrounded by the same element
                v = w[e_i-1]
            else:
                v = elem
        elif (gc == gc_max).sum() > 1:
            # tie between labels
            if gc_max == glob_elem_count:
                # we keep it
                v = elem
            else:
                # first max
                v = w_hat[gc_max_i]
        else:
            v = w_hat[gc_max_i]
    else:
        if elem_count == c_max:
            v = elem
        else:
            v = w[c_max_i]
    return v


def label_filter(
        labels: np.ndarray,
        min_repetition: int,
        label_undecidable: bool = True,
        relabel_output: bool = True
) -> np.ndarray:
    if min_repetition == 1:
        return labels
    glob_counts = np.r_[np.bincount(labels), 0]  # for -1 labels
    l2d, c2d, flagged = _get_counts(labels, min_repetition)
    while np.any(flagged):
        out = np.zeros_like(labels)
        for i in flagged.nonzero()[0]:
            out[i] = _filter_window(l2d[i], c2d[i], glob_counts, label_undecidable)
        if np.all(labels[flagged] == out[flagged]):
            break
        out[~flagged] = labels[~flagged]
        labels = out
        l2d, c2d, flagged = _get_counts(labels, min_repetition)
    if relabel_output:
        _, labels = np.unique(labels, return_inverse=True)
    return labels


@dtc.dataclass
class LabelFilter(Functional):
    min_repetition: int = 1
    label_undecidable: bool = False

    def np_func(self, inputs):
        return label_filter(inputs,
                            self.min_repetition,
                            self.label_undecidable,
                            relabel_output=False)

    def torch_func(self, inputs):
        pass

    @property
    def inv(self) -> "Functional":
        return None
