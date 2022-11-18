def demo():
    """### Load Data"""
    import mimikit as mmk
    import h5mapper as h5m
    import os


    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = ['./data']
    # audio sample rate
    sr = 22050
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
        sr=SoundBank.snd.sr,
        n_fft=n_fft,
        hop_length=hop_length,
        coordinate='mag',
        center=False
    )

    # NETWORK

    net = mmk.WaveNetFFT(
        feature=feature,
        input_heads=1,
        output_heads=1,
        scaled_activation=False,

        # number of layers (per block)
        blocks=(4,),
        # dimension of the layers
        dims_dilated=(1024,),
        groups=2,

    )
    net.use_fast_generate = False

    # OPTIMIZATION LOOP

    mmk.train(
        soundbank,
        net,
        root_dir="./",
        input_feature=feature,
        target_feature=feature,

        # BATCH

        batch_size=4,
        batch_length=64,
        downsampling=feature.hop_length // 1,
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
        OUTPUT_TRAINING="mp3",

        every_n_epochs=10,
        n_examples=4,
        prompt_length=64,
        n_steps=int(12 * (feature.sr // feature.hop_length)),
    )

    """----------------------------"""
