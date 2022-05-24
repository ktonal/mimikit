

def demo():
    """Generate From Checkpoint"""
    import mimikit as mmk
    import h5mapper as h5m
    import torch
    import matplotlib.pyplot as plt
    import librosa

    # load a checkpoint
    ckpt = mmk.Checkpoint(
        root_dir="./trainings/wn-test-gesten",
        id='84e89798ec2c85e19790344fb598932118c7a65142e747e383907c5f7ced0f26',
        epoch=1
    )
    net, feature = ckpt.network, ckpt.feature

    # prompt positions in seconds
    indices = [
        1.1, 8.5, 46.3
    ]
    # duration in seconds to generate converted to number of steps
    n_steps = librosa.time_to_frames(8, sr=feature.sr, hop_length=feature.hop_length)

    class SoundBank(h5m.TypedFile):
        snd = h5m.Sound(sr=feature.sr, mono=True, normalize=True)

    SoundBank.create("gen.h5", ckpt.train_hp["files"], )
    soundbank = SoundBank("gen.h5", mode='r', keep_open=True)

    def process_outputs(outputs, bidx):
        output = feature.inverse_transform(outputs[0])
        for i, out in enumerate(output):
            y = out.detach().cpu().numpy()
            plt.figure(figsize=(20, 2))
            plt.plot(y)
            plt.show(block=False)
            mmk.audio(y, sr=feature.sr,
                      hop_length=feature.hop_length)

    max_i = soundbank.snd.shape[0] - getattr(feature, "hop_length", 1) * net.rf
    g_dl = soundbank.serve(
        (feature.batch_item(shift=0, length=net.rf, training=False),),
        sampler=mmk.IndicesSampler(N=len(indices),
                                   indices=[librosa.time_to_samples(i, sr=feature.sr) for i in indices],
                                   max_i=max_i,
                                   redraw=False),
        shuffle=False,
        batch_size=len(indices)
    )

    loop = mmk.GenerateLoop(
        network=net,
        dataloader=g_dl,
        inputs=(h5m.Input(None, h5m.AsSlice(dim=1, shift=-net.rf, length=net.rf), setter=h5m.Setter(dim=1)),),
        n_steps=n_steps,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        time_hop=net.hp.get("hop", 1),
        process_outputs=process_outputs
    )
    loop.run()

    """----------------------------"""
