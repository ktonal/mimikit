def demo():
    """### Load Data"""
    import mimikit as mmk
    import h5mapper as h5m
    import os

    # DATA

    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = tuple(h5m.FileWalker(mmk.SOUND_FILE_REGEX, "./"))
    SAMPLE_RATE = 22050

    db_path = "train-freqnet.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    signal = mmk.Extractor(
        "signal",
        mmk.Compose(mmk.FileToSignal(SAMPLE_RATE), mmk.RemoveDC(), mmk.Normalize()))
    ds = mmk.DatasetConfig(sources=sources,
                           filename=db_path,
                           extractors=(signal,))
    ds.create(mode="w")
    dataset = ds.get(mode="r", keep_open=True)

    N = dataset.signal.shape[0]
    print(f"Dataset length in minutes is: {(N / SAMPLE_RATE) / 60:.2f}")
    print("Extracted following files:")
    for f in dataset.index:
        print("\t", f)

    """### Configure Network"""

    # INPUT / TARGET
    io = mmk.IOSpec.magspec_io(
        mmk.IOSpec.MagSpecIOConfig(
            sr=SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            activation="Identity"
        ),
        signal
    )

    # NETWORK

    net = mmk.WaveNet.from_config(
        mmk.WaveNet.Config(
            io_spec=io,
            kernel_sizes=(2,),
            blocks=(3,),
            dims_dilated=(2048,),
            apply_residuals=False,
            residuals_dim=None,
            skips_dim=None,
            groups=8,
            act_f="Tanh",
            act_g="Sigmoid",
            pad_side=0,
            bias=True,
            use_fast_generate=False,
            tie_io_weights=False
        ))

    """### Configure Training"""

    # OPTIMIZATION LOOP
    loop = mmk.TrainARMLoop.from_config(
        mmk.TrainARMConfig(max_lr=1e-3,
                           betas=(0.9, 0.9),
                           div_factor=1.,
                           final_div_factor=1.,
                           pct_start=0.0,
                           n_examples=4,
                           prompt_length_sec=3.,
                           batch_size=16,
                           tbptt_chunk_length=None,
                           batch_length=64,
                           downsampling=64,
                           limit_train_batches=10000,
                           max_epochs=300,
                           every_n_epochs=10,
                           outputs_duration_sec=60,
                           MONITOR_TRAINING=True,
                           OUTPUT_TRAINING=False,
                           CHECKPOINT_TRAINING=True),
        dataset,
        net,
    )

    """### RUN"""

    loop.run()
    None

    """----------------------------"""
