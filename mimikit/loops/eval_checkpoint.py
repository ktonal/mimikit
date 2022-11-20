import h5mapper as h5m
import torch
import numpy as np

from . import IndicesSampler, GenerateLoop
from ..models import Checkpoint
from ..extract.from_neighbors import nearest_neighbor, cum_entropy
from ..utils import audio

__all__ = [
    "eval_checkpoint"
]


def eval_checkpoint(ckpt: Checkpoint, soundbank: h5m.SoundBank):
    net = ckpt.network
    feature = ckpt.feature
    saved = {}

    def process_outputs(outputs, bidx):
        outputs = outputs[0]
        y = feature.transform(soundbank.snd[:])
        y = torch.from_numpy(y).to(outputs)
        nn = torch.stack([nearest_neighbor(out, y)[1] for out in outputs])
        hx = torch.stack([cum_entropy(n, neg_diff=False) for n in nn]).detach().cpu().numpy()
        idx = np.argsort(hx)
        for i in idx:
            saved[hx[i]] = outputs[i].detach().cpu().numpy().T
        del y
        del nn
        torch.cuda.empty_cache()

    prompt_files = soundbank
    batch_item = feature.batch_item(shift=0, length=net.rf, training=False)
    indices = IndicesSampler(
        N=500,
        indices=torch.arange(
            0,
            prompt_files.snd.shape[0] - batch_item.getter.length,
            (prompt_files.snd.shape[0] - batch_item.getter.length) // 500))
    dl = prompt_files.serve(
        (batch_item,),
        sampler=indices,
        shuffle=False,
        batch_size=64,
    )

    loop = GenerateLoop(
        network=net,
        dataloader=dl,
        inputs=(h5m.Input(None,
                          getter=h5m.AsSlice(dim=1, shift=-net.rf, length=net.rf),
                          setter=h5m.Setter(dim=1)),),
        n_steps=feature.sr * 25 // feature.hop_length,
        add_blank=True,
        time_hop=net.hp.get("hop", 1),
        process_outputs=process_outputs
    )
    print("\n")
    print("\n")
    print("-----------------------------------------")
    print("\n")
    print("\n")
    loop.run()
    print("\n")
    print("\n")
    print(ckpt)

    for k in list(sorted(saved))[-8:]:
        print("SCORE = ", k)
        audio(saved[k], hop_length=feature.hop_length, sr=feature.sr)
