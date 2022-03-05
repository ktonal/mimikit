def demo():
    """### Configure and run training"""
    import mimikit as mmk
    import h5mapper as h5m
    import torch
    import os

    # DATA

    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = ['./data']
    # audio sample rate
    sr = 16000

    db_path = "train.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    class SoundBank(h5m.TypedFile):
        snd = h5m.Sound(sr=sr, mono=True, normalize=True)

    SoundBank.create(db_path, sources)
    soundbank = SoundBank(db_path, mode='r', keep_open=True)

    # INPUT / TARGET

    feature = mmk.MuLawSignal(
        sr=soundbank.snd.sr,
        q_levels=256,
    )

    # NETWORK

    net = mmk.models.SampleRNN(
        feature=feature,
        chunk_length=16000 * 8,
        frame_sizes=(64, 16, 16),
        dim=512,
        n_rnn=2,
        q_levels=feature.q_levels,
        embedding_dim=256,
        mlp_dim=512,
    )

    # OPTIMIZATION LOOP

    mmk.train(
        soundbank,
        net,
        input_feature=mmk.MultiScale(feature, net.frame_sizes, (*net.frame_sizes[:-1], 1)),
        target_feature=feature,
        root_dir="./",

        # BATCH

        batch_size=16,
        batch_length=1024,
        oversampling=2,
        shift_error=0,
        tbptt_chunk_length=16000 * 8,

        # OPTIM

        max_epochs=200,
        limit_train_batches=None,

        max_lr=5e-4,
        betas=(0.9, 0.93),
        div_factor=20.,
        final_div_factor=1.,
        pct_start=0.,
        cycle_momentum=False,
        reset_optim=True,

        # MONITORING / OUTPUTS

        CHECKPOINT_TRAINING=False,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING="",

        every_n_epochs=10,
        n_examples=4,
        prompt_length=16000,
        n_steps=int(6 * (feature.sr)),
        temperature=torch.tensor([[.999], [1.25], [1.05], [.9]]).repeat(1, int(6 * (feature.sr))),
    )

    """----------------------------"""
