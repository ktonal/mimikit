def demo():
    """Generate From Checkpoint"""
    import mimikit as mmk
    import torch

    # load a checkpoint
    ckpt = mmk.Checkpoint(
        root_dir="./",
        id='84e89798e',
        epoch=1
    )
    N_TRIALS = 500
    K_BESTS = 10

    dataset, network = ckpt.dataset, ckpt.network
    S = torch.from_numpy(network.config.io_spec.inputs[0].transform(
        dataset.signal[:]
    )).to(mmk.default_device())
    # prompt positions in seconds

    loop = mmk.GenerateLoopV2.from_config(
        mmk.GenerateLoopV2.Config(
            output_duration_sec=30.,
            prompts_length_sec=1.,
            prompts_position_sec=(
                1.1, 8.5, 46.3
            ),
            batch_size=32,
            display_waveform=False,
            yield_inversed_outputs=True
        ),
        dataset,
        network
    )
    saved = {}
    for outputs in loop.run():
        outputs = outputs[0]
        with torch.no_grad():
            nn = torch.stack([mmk.nearest_neighbor(out, S)[1] for out in outputs])
            hx = torch.stack([mmk.cum_entropy(n, neg_diff=False) for n in nn])
            idx = torch.argsort(hx).detach().cpu().numpy()
        # TODO: only keep k_bests outputs
        for i in idx:
            saved[hx[i]] = outputs[i].detach().cpu().numpy()
        del nn
        torch.cuda.empty_cache()
    # todo display k_bests

    """----------------------------"""