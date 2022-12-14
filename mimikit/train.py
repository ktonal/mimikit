import json
import os
import hashlib

import h5mapper as h5m
import torch
import dataclasses as dtc

from .features import AudioSignal
from .loops import GenerateCallback, GenerateLoop, AudioLogger, MMKCheckpoint, TBPTTSampler, IndicesSampler, TrainLoop

__all__ = [
    "train"
]

from .loops.train_loops import TrainARMConfig


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super(MyEncoder, self).default(obj)
        except:
            return str(obj)


def train(
        cfg: TrainARMConfig,
        model_config,
        soundbank: h5m.SoundBank,
        net,
        input_feature=AudioSignal(sr=22050),
        target_feature=AudioSignal(sr=22050),
        # additional hyper-params:
        **kwargs
):
    train_hp = dtc.asdict(cfg)
    train_hp["temperature"] = cfg.temperature[0] if cfg.temperature is not None else cfg.temperature
    hp = dict(files=list(soundbank.index.keys()),
              network_class=net.__class__.__qualname__,
              network=net.hp,
              **kwargs,
              train_hp=train_hp)
    ID = hashlib.sha256(json.dumps(hp, cls=MyEncoder).encode("utf-8")).hexdigest()
    print("****************************************************")
    print("ID IS :", ID)
    print("****************************************************")
    hp['id'] = ID
    root_dir = os.path.join(cfg.root_dir, ID)
    os.makedirs(root_dir, exist_ok=True)
    if "mp3" in cfg.OUTPUT_TRAINING:
        output_dir = os.path.join(root_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        filename_template = os.path.join(output_dir, "epoch{epoch}_prm{prompt_idx}.wav")
    else:
        filename_template = ""

    with open(os.path.join(root_dir, "hp.json"), "w") as fp:
        json.dump(hp, fp, cls=MyEncoder)

    # DATALOADER

    batch = (
        input_feature.batch_item(
            data=torch.from_numpy(soundbank.snd[:]) if cfg.in_mem_data else "snd",
            shift=0, length=cfg.batch_length, downsampling=cfg.downsampling),
        target_feature.batch_item(
            data=torch.from_numpy(soundbank.snd[:]) if cfg.in_mem_data else "snd",
            shift=net.shift, length=net.output_length(cfg.batch_length),
            downsampling=cfg.downsampling)
    )

    if cfg.tbptt_chunk_length is not None:
        if getattr(input_feature, 'domain', '') == 'time-freq' and isinstance(getattr(batch[0], 'getter', False),
                                                                              h5m.Getter):
            item_len = batch[0].getter.length
            seq_len = item_len
            chunk_length = item_len * cfg.tbptt_chunk_length
            N = len(h5m.ProgrammableDataset(soundbank, batch))
        else:  # feature is in time domain
            seq_len = cfg.batch_length
            chunk_length = cfg.tbptt_chunk_length
            N = soundbank.snd.shape[0]
        loader_kwargs = dict(
            batch_sampler=TBPTTSampler(
                N,
                batch_size=cfg.batch_size,
                chunk_length=chunk_length,
                seq_len=seq_len,
                oversampling=cfg.oversampling
            )
        )
    else:
        loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=True)
    n_workers = max(os.cpu_count(), min(cfg.batch_size, os.cpu_count()))
    prefetch = (cfg.batch_size // n_workers) // 2
    dl = soundbank.serve(batch,
                         num_workers=n_workers,
                         prefetch_factor=max(prefetch, 1),
                         pin_memory=True,
                         persistent_workers=True,
                         **loader_kwargs
                         )

    opt = torch.optim.Adam(net.parameters(), lr=cfg.max_lr, betas=cfg.betas)
    sched = torch.OneCycleLR(
        opt,
        steps_per_epoch=min(len(dl), cfg.limit_train_batches) if cfg.limit_train_batches is not None else len(dl),
        epochs=cfg.max_epochs,
        max_lr=cfg.max_lr,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
        pct_start=cfg.pct_start,
        cycle_momentum=cfg.cycle_momentum
    )
    # TrainLoop
    tr_loop = TrainLoop(
        train_config=model_config,
        loader=dl,
        net=net,
        loss_fn=target_feature.loss_fn,
        optim=([opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]),
        tbptt_len=cfg.tbptt_chunk_length // cfg.batch_length if cfg.tbptt_chunk_length is not None else None,
    )

    # Gen Loop
    max_i = soundbank.snd.shape[0] - getattr(input_feature, "hop_length", 1) * cfg.prompt_length
    g_dl = soundbank.serve(
        (input_feature.batch_item(shift=0, length=cfg.prompt_length, training=False),),
        sampler=IndicesSampler(N=cfg.n_examples,
                               max_i=max_i,
                               redraw=True),
        shuffle=False,
        batch_size=cfg.n_examples
    )
    gen_loop = GenerateLoop(
        network=net,
        dataloader=g_dl,
        inputs=(h5m.Input(None, h5m.AsSlice(dim=1, shift=-net.rf, length=net.rf),
                          setter=h5m.Setter(dim=1)),
                *((h5m.Input(cfg.temperature, h5m.AsSlice(dim=1 + int(hasattr(net.hp, 'hop')), length=1),
                             setter=None),)
                  if cfg.temperature is not None else ())),
        n_steps=cfg.n_steps,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        time_hop=net.hp.get("hop", 1)
    )

    callbacks = []

    if cfg.CHECKPOINT_TRAINING:
        callbacks += [
            MMKCheckpoint(epochs=cfg.every_n_epochs, root_dir=root_dir)
        ]

    if cfg.MONITOR_TRAINING or cfg.OUTPUT_TRAINING:
        callbacks += [
            GenerateCallback(
                generate_loop=gen_loop,
                every_n_epochs=cfg.every_n_epochs,
                output_features=[target_feature, ],
                audio_logger=AudioLogger(
                    sr=target_feature.sr,
                    hop_length=getattr(target_feature, 'hop_length', 512),
                    **(dict(filename_template=filename_template,
                            target_dir=os.path.dirname(filename_template))
                       if 'mp3' in cfg.OUTPUT_TRAINING else {}),
                ),
            )]

    gen_loop.plot_audios = gen_loop.play_audios = cfg.MONITOR_TRAINING

    tr_loop.run(
        root_dir=cfg.root_dir,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        limit_train_batches=cfg.limit_train_batches if cfg.limit_train_batches is not None else 1.
    )
    try:
        dl._iterator._shutdown_workers()
    except:
        pass
    soundbank.close()
    return tr_loop, net
