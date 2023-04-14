def demo():
    """### imports"""

    import h5mapper as h5m
    import mimikit as mmk
    from pbind import Pseq, Pbind, Pwhite, inf

    """### Get some checkpoints"""
    ROOT_DIR = './'
    checkpoints = {}
    for i, path in enumerate(h5m.FileWalker(mmk.CHECKPOINT_REGEX, ROOT_DIR)):
        checkpoints[i] = mmk.Checkpoint.from_path(path)
    checkpoints

    """### Get the prompts from which to generate"""
    db = checkpoints[0].dataset

    OUTPUT_SR = 22050
    PROMPTS_POS_SEC = (
        0, OUTPUT_SR // 2, OUTPUT_SR
    )
    PROMPT_LENGTH_SEC = OUTPUT_SR

    # get a batch of prompts
    prompts = next(iter(db.serve(
        (h5m.Input(data='signal', getter=h5m.AsSlice(shift=0, length=PROMPT_LENGTH_SEC)),),
        shuffle=False,
        # batch_size=1 --> new stream for each prompt <> batch_size=8 --> one stream for 8 prompts :
        batch_size=len(PROMPTS_POS_SEC),
        sampler=mmk.IndicesSampler(
            # INDICES FOR THE PROMPTS :
            indices=PROMPTS_POS_SEC
        ))))[0]
    prompts.shape

    """### Define a pattern of models"""

    # THE MODELS PATTERN defines which checkpoint (id, epoch) generates for how long (seconds)

    stream = Pseq([
        Pbind(
            "generator", checkpoints[0],
            "seconds", Pwhite(lo=3., hi=5., repeats=1)
        ),
        # Pbind(
        #     # TODO: This event inserts the most similar continuation from the Trainset "Cough"
        #     "seconds", Pwhite(lo=2., hi=5., repeats=1)
        # ),
        Pbind(
            "generator", checkpoints[1],
            # SampleRNN Checkpoints work best with a temperature parameter :
            "temperature", Pwhite(lo=.25, hi=1.5),
            "seconds", Pwhite(lo=.1, hi=1., repeats=1),
        )
    ], inf).asStream()
    stream

    """### Generate"""

    TOTAL_SECONDS = 10.

    ensemble = mmk.EnsembleGenerator(
        prompts, TOTAL_SECONDS, OUTPUT_SR, stream,
        # with this you can print the event -- or not
        print_events=False
    )
    outputs = ensemble.run()
    logger = mmk.AudioLogger(sr=OUTPUT_SR)
    logger.display_batch(outputs)


    """----------------------------"""