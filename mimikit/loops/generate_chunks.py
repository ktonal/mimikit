def main():
    import mimikit as mmk
    import h5mapper as h5m
    import numpy as np

    # load a checkpoint
    ckpt = mmk.Checkpoint(
        root_dir="../../../trainings",
        id='srnn_1min_chunk',
        epoch=20
    )
    # create output array
    tf = h5m.TypedFile("srnn_1min_chunk_outputs.h5", mode="w")

    # get the prompts and the loop's config
    bs = 64
    positions = np.random.choice(np.linspace(10, 3600, 3000), size=bs)
    temperature = dict(temperature=np.random.choice(np.linspace(.85, .999, bs * 4), size=bs, replace=False))
    config = mmk.GenerateLoopV2.Config(
        output_duration_sec=30.,
        prompts_length_sec=5.,
        prompts_position_sec=(
            *positions,
        ),
        batch_size=bs,
        display_waveform=False,
        yield_inversed_outputs=False,
        parameters=temperature
    )

    seed = next(iter(mmk.GenerateLoopV2.get_dataloader(config, ckpt.dataset, ckpt.network)))
    # store the prompts (TODO: Transpose!)
    tf.add("0", {"outputs": {str(i): x for i, x in enumerate(seed[1].cpu().numpy())}})
    feature = ckpt.network_config.io_spec.targets[0]
    sr = feature.sr

    # generate by chunk
    for i in mmk.tqdm(range(1, 10)):
        tracks = tf.outputs.get(str(i - 1))
        loader = [[np.ones(1), np.stack([tracks[i][-int(sr * config.prompts_length_sec):]
                                         for i in sorted(tracks, key=int)])]]
        config.parameters["temperature"] += np.random.randn(bs) * .1
        config.parameters["temperature"] = np.maximum(np.minimum(config.parameters["temperature"], .999), .85)
        loop = mmk.GenerateLoopV2(
            config,
            ckpt.network,
            int(sr * config.output_duration_sec),
            loader,
        )
        for output in loop.run():
            tf.outputs.add(
                str(i),
                {str(i): x for i, x in
                 enumerate(output[0][:, int(sr * config.prompts_length_sec):].detach().cpu().numpy())})
            break
        tf.flush()

    """ display the result """

    feature = ckpt.network_config.io_spec.targets[0]
    logger = mmk.AudioLogger(sr=feature.sr,
                             title_template="track={track} - position={position} - temperature={temperature}")

    for i in range(bs):
        track = getattr(tf.outputs, str(i))[:]
        track = feature.inv(track)
        logger.display(track, track=str(i), position=positions[i], temperature=temperature["temperature"][i])