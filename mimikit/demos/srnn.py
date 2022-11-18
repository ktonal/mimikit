def demo():
    """### Load Data"""
    import mimikit as mmk
    import h5mapper as h5m
    import torch
    import os


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

    net = mmk.models.SampleRNN(
        feature=feature,
        # bottom tier has linearized Mu-Law signal as input (no embeddings!)
        input_type="lin",
        # output MLP learns to predict the temperature of its output distribution
        learn_temperature=True,
        chunk_length=16000 * 8,
        frame_sizes=(256, 128, 64, 32, 16, 8, 4, 4),
        dim=128,
        n_rnn=1,
        q_levels=feature.q_levels,
        embedding_dim=1,
        mlp_dim=128,
    )

    # OPTIMIZATION LOOP

    mmk.train(
        soundbank,
        net,
        input_feature=mmk.MultiScale(feature, net.frame_sizes, (*net.frame_sizes[:-1], 1)),
        target_feature=feature,
        root_dir="./",

        # BATCH

        batch_size=32,
        batch_length=2048,
        oversampling=16,
        shift_error=0,
        tbptt_chunk_length=net.hp.chunk_length,

        # OPTIM

        max_epochs=1000,
        limit_train_batches=None,

        max_lr=1e-3,
        betas=(0.9, 0.9),
        div_factor=1.,
        final_div_factor=10.,
        pct_start=0.05,
        cycle_momentum=False,
        reset_optim=False,

        # MONITORING / OUTPUTS

        CHECKPOINT_TRAINING=False,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING="",

        every_n_epochs=100,
        n_examples=4,
        # small warm up prompts work better than big ones...
        prompt_length=net.frame_sizes[0]*6,
        n_steps=int(16 * feature.sr),
        temperature=torch.tensor([[1.], [1.5], [.5], [.05]]).repeat(1, int(16*net.feature.sr)),

    )

    """----------------------------"""
