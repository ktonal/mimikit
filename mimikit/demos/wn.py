def demo():
    """### Load Data"""
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

    """### Configure and run training"""

    # INPUT / TARGET

    feature = mmk.MuLawSignal(
        sr=soundbank.snd.sr,
        q_levels=256,
    )

    # NETWORK

    net = mmk.WaveNetQx(
        feature=feature,
        mlp_dim=1024,

        kernel_sizes=(8, 8, 4, 2),
        blocks=(4,),
        dims_dilated=(1024,),
        dims_1x1=(),
        residuals_dim=None,
        apply_residuals=False,
        skips_dim=None,
        groups=8,
        pad_side=0,
        stride=1,
        bias=True,
    )
    net.use_fast_generate = True

    # OPTIMIZATION LOOP

    mmk.train(
        soundbank,
        net,
        root_dir="./trainings/wn-legacy-test",
        input_feature=feature,
        target_feature=feature,

        # BATCH

        batch_size=16,
        batch_length=2048,
        downsampling=8,
        shift_error=0,

        # OPTIM

        max_epochs=200,
        limit_train_batches=1000,

        max_lr=1e-3,
        betas=(0.91, 0.95),
        div_factor=1.,
        final_div_factor=3.,
        pct_start=0.,
        cycle_momentum=False,

        # MONITORING / OUTPUTS

        CHECKPOINT_TRAINING=False,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING="",

        every_n_epochs=10,
        n_examples=4,
        prompt_length=net.rf,
        n_steps=int(4 * feature.sr),
        temperature=torch.tensor([[2.], [1.], [.9], [.5]]).repeat(1, int(4 * (feature.sr))),
    )