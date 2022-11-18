def demo():
    """### Load Data"""
    import mimikit as mmk
    import h5mapper as h5m
    import os


    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = ['./data']
    # audio sample rate
    sr = 44100
    # the size of the stft
    n_fft = 2048
    # hop_length of the stft
    hop_length = n_fft // 4

    db_path = "train.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    class SoundBank(h5m.TypedFile):
        snd = h5m.Sound(sr=sr, mono=True, normalize=True)

    SoundBank.create(db_path, sources)
    soundbank = SoundBank(db_path, mode='r', keep_open=True)

    """### Configure and run training"""

    # INPUT / TARGET

    feature = mmk.Spectrogram(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        coordinate='mag',
        center=False
    )

    net = mmk.Seq2SeqLSTM(
        feature=feature,
        input_heads=1,
        output_heads=1,
        scaled_activation=True,
        model_dim=512,
        num_layers=1,
        n_lstm=1,
        bottleneck="add",
        n_fc=1,
        hop=4,
        weight_norm=False,
        with_tbptt=False,
        with_sampler=True,
    )

    mmk.train(
        soundbank,
        net,
        root_dir="./",
        input_feature=feature,
        target_feature=feature,

        # BATCH

        batch_size=16,
        batch_length=net.hp.hop,
        downsampling=feature.hop_length // 8,
        shift_error=0,

        # OPTIM

        max_epochs=100,
        limit_train_batches=None,

        max_lr=1e-3,
        betas=(0.9, 0.9),
        div_factor=3.,
        final_div_factor=1.,
        pct_start=0.,
        cycle_momentum=False,
        reset_optim=False,

        # MONITORING / OUTPUTS

        CHECKPOINT_TRAINING=False,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING="",

        every_n_epochs=10,
        n_examples=4,
        prompt_length=net.hp.hop,
        n_steps=int(16 * (feature.sr // feature.hop_length // net.hp.hop)),
    )

    """----------------------------"""
