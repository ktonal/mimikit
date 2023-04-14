def demo():
    """### Load Data"""
    import mimikit as mmk
    import h5mapper as h5m
    import os

    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = tuple(h5m.FileWalker(mmk.SOUND_FILE_REGEX, "./"))
    SAMPLE_RATE = 16000

    db_path = "train-srnn.h5"
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

    io = mmk.IOSpec.mulaw_io(extractor=signal,
                             config=mmk.IOSpec.MuLawIOConfig(
                                 sr=SAMPLE_RATE,
                                 compression=.5,
                                 mlp_dim=128,
                                 n_mlp_layers=0,
                                 min_temperature=1e-3
                             ))

    # NETWORK

    net = mmk.SampleRNN.from_config(
        mmk.SampleRNN.Config(rnn_class="lstm",
                             n_rnn=1,
                             rnn_dropout=0.0,
                             frame_sizes=(256, 128, 64, 32, 16, 8, 4, 8),
                             hidden_dim=128,
                             weight_norm=True,
                             io_spec=io))

    """### Configure Training"""

    # OPTIMIZATION LOOP
    loop = mmk.TrainARMLoop.from_config(
        mmk.TrainARMConfig(max_lr=1e-3,
                           betas=(0.9, 0.9),
                           div_factor=1.,
                           final_div_factor=1.,
                           pct_start=0.0,
                           temperature=(1., .75, 0.5, .1),
                           n_examples=4,
                           prompt_length_sec=1.,
                           batch_size=32,
                           tbptt_chunk_length=8 * SAMPLE_RATE,
                           batch_length=2048,
                           oversampling=4,
                           limit_train_batches=None,
                           max_epochs=2000,
                           every_n_epochs=5,
                           outputs_duration_sec=10,
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
