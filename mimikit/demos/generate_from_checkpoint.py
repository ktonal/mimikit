def demo():
    """Generate From Checkpoint"""
    import mimikit as mmk

    # load a checkpoint
    ckpt = mmk.Checkpoint(
        root_dir="./",
        id='84e89798e',
        epoch=1
    )

    # prompt positions in seconds

    loop = mmk.GenerateLoopV2.from_config(
        mmk.GenerateLoopV2.Config(
            output_duration_sec=30.,
            prompts_length_sec=1.,
            prompts_position_sec=(
                1.1, 8.5, 46.3
            ),
            batch_size=3,
            display_waveform=True
        ),
        ckpt.dataset,
        ckpt.network
    )
    for _ in loop.run():
        continue

    """----------------------------"""
