import json
import os
import hashlib
import h5mapper as h5m
import torch

from .features import AudioSignal
from .loops import GenerateCallback, GenerateLoop, AudioLogger, MMKCheckpoint, TBPTTSampler, IndicesSampler, TrainLoop

__all__ = [
    "train"
]


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super(MyEncoder, self).default(obj)
        except:
            return str(obj)


def train(
        soundbank: h5m.SoundBank,
        net,
        input_feature=AudioSignal(sr=22050),
        target_feature=AudioSignal(sr=22050),
        root_dir='./trainings',
        batch_size=16,
        batch_length=32,
        downsampling=1,
        shift_error=0,
        tbptt_chunk_length=None,
        oversampling=1,

        max_epochs=2,
        limit_train_batches=1000,
        max_lr=5e-4,
        betas=(0.9, 0.93),
        div_factor=3.,
        final_div_factor=1.,
        pct_start=0.,
        cycle_momentum=False,
        reset_optim=False,

        CHECKPOINT_TRAINING=True,

        MONITOR_TRAINING=True,
        OUTPUT_TRAINING='',

        every_n_epochs=2,
        n_examples=3,
        prompt_length=32,
        n_steps=200,
        temperature=None,
        # additional hyper-params:
        **kwargs
):
    train_hp = dict(locals())
    train_hp.pop("net")
    train_hp.pop("soundbank")
    train_hp.pop("root_dir")
    train_hp.pop("kwargs")
    train_hp["temperature"] = temperature[0] if temperature is not None else temperature
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
    root_dir = os.path.join(root_dir, ID)
    os.makedirs(root_dir, exist_ok=True)
    if "mp3" in OUTPUT_TRAINING:
        output_dir = os.path.join(root_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        filename_template = os.path.join(output_dir, "epoch{epoch}_prm{prompt_idx}.wav")
    else:
        filename_template = ""

    with open(os.path.join(root_dir, "hp.json"), "w") as fp:
        json.dump(hp, fp, cls=MyEncoder)
    if "h5" in OUTPUT_TRAINING or CHECKPOINT_TRAINING:
        logs_file = os.path.join(root_dir, "checkpoints.h5")
    else:
        logs_file = None

    # DATALOADER
    batch = (
        input_feature.batch_item(shift=0, length=batch_length, downsampling=downsampling),
        target_feature.batch_item(shift=net.shift, length=net.output_length(batch_length), downsampling=downsampling)
    )

    if tbptt_chunk_length is not None:
        if getattr(input_feature, 'domain', '') == 'time-freq' and isinstance(getattr(batch[0], 'getter', False),
                                                                              h5m.Getter):
            item_len = batch[0].getter.length
            seq_len = item_len
            chunk_length = item_len * tbptt_chunk_length
            N = len(h5m.ProgrammableDataset(soundbank, batch))
        else:  # feature is in time domain
            seq_len = batch_length
            chunk_length = tbptt_chunk_length
            N = soundbank.snd.shape[0]
        loader_kwargs = dict(
            batch_sampler=TBPTTSampler(
                N,
                batch_size=batch_size,
                chunk_length=chunk_length,
                seq_len=seq_len,
                oversampling=oversampling
            )
        )
    else:
        loader_kwargs = dict(batch_size=batch_size, shuffle=True)
    n_workers = max(os.cpu_count(), min(batch_size, os.cpu_count()))
    prefetch = (batch_size // n_workers) // 2
    dl = soundbank.serve(batch,
                         num_workers=n_workers,
                         prefetch_factor=prefetch,
                         pin_memory=True,
                         # True leads to memory leaks, False resets the processes at each epochs
                         persistent_workers=False,
                         **loader_kwargs
                         )

    opt = torch.optim.Adam(net.parameters(), lr=max_lr, betas=betas)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        steps_per_epoch=min(len(dl), limit_train_batches) if limit_train_batches is not None else len(dl),
        epochs=max_epochs,
        max_lr=max_lr,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        pct_start=pct_start,
        cycle_momentum=cycle_momentum
    )
    # TrainLoop
    tr_loop = TrainLoop(
        loader=dl,
        net=net,
        loss_fn=target_feature.loss_fn,
        optim=([opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]),
        tbptt_len=tbptt_chunk_length // batch_length if tbptt_chunk_length is not None else None,
        reset_optim=reset_optim,
    )

    # Gen Loop
    max_i = soundbank.snd.shape[0] - getattr(input_feature, "hop_length", 1) * prompt_length
    g_dl = soundbank.serve(
        (input_feature.batch_item(shift=0, length=prompt_length, training=False),),
        sampler=IndicesSampler(N=n_examples,
                               max_i=max_i,
                               redraw=True),
        shuffle=False,
        batch_size=n_examples
    )
    gen_loop = GenerateLoop(
        network=net,
        dataloader=g_dl,
        inputs=(h5m.Input(None, h5m.AsSlice(dim=1, shift=-net.rf, length=net.rf),
                          setter=h5m.Setter(dim=1)),
                *((h5m.Input(temperature, h5m.AsSlice(dim=1 + int(hasattr(net.hp, 'hop')), length=1),
                             setter=None),)
                  if temperature is not None else ())),
        n_steps=n_steps,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        time_hop=net.hp.get("hop", 1)
    )

    # class Logs(h5m.TypedFile):
    #     ckpt = h5m.TensorDict(net.state_dict()) if CHECKPOINT_TRAINING else None
    #     outputs = h5m.Array() if 'h5' in OUTPUT_TRAINING else None

    # if "h5" in OUTPUT_TRAINING or CHECKPOINT_TRAINING:
    #     logs = Logs(logs_file, mode='w')
    # else:
    #     logs = None

    callbacks = []

    if CHECKPOINT_TRAINING:
        callbacks += [
            MMKCheckpoint(epochs=every_n_epochs, root_dir=root_dir)
        ]

    if MONITOR_TRAINING or OUTPUT_TRAINING:
        callbacks += [
            GenerateCallback(
                generate_loop=gen_loop,
                every_n_epochs=every_n_epochs,
                output_features=[target_feature, ],
                audio_logger=AudioLogger(
                    sr=target_feature.sr,
                    hop_length=getattr(target_feature, 'hop_length', 512),
                    **(dict(filename_template=filename_template,
                            target_dir=os.path.dirname(filename_template))
                       if 'mp3' in OUTPUT_TRAINING else {}),
                    # **(dict(id_template="idx_{prompt_idx}",
                    #         proxy_template="outputs/epoch_{epoch}/",
                    #         target_bank=logs)
                    #    if 'h5' in OUTPUT_TRAINING else {})
                ),
            )]

    gen_loop.plot_audios = gen_loop.play_audios = MONITOR_TRAINING

    tr_loop.run(max_epochs=max_epochs,
                logger=None,
                callbacks=callbacks,
                limit_train_batches=limit_train_batches if limit_train_batches is not None else 1.
                )
    try:
        dl._iterator._shutdown_workers()
    except:
        pass
    soundbank.close()
    return tr_loop, net
