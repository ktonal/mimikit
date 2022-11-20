def demo():

    """# !! DON'T FORGET TO `%pip install pypbind` FIRST !!"""

    import h5mapper as h5m
    import mimikit as mmk
    from pbind import Pseq, Pbind, Prand, Pwhite, inf

    BASE_SR = 22050

    # get a soundbank first!

    class SoundBank(h5m.TypedFile):
        snd = h5m.Sound(sr=BASE_SR, mono=True, normalize=True)

    soundbank = SoundBank.create("ensemble.h5", "./")
    soundbank.info()
    soundbank.index

    """### Define the prompts from which to generate"""

    prompts = soundbank.serve(
        (h5m.Input(data='snd', getter=h5m.AsSlice(shift=0, length=BASE_SR)),),
        shuffle=False,
        # batch_size=1 --> new stream for each prompt <> batch_size=8 --> one stream for 8 prompts :
        batch_size=1,
        sampler=mmk.IndicesSampler(
            # INDICES FOR THE PROMPTS :
            indices=(0, BASE_SR * 8, BASE_SR * 16)
        ))

    """### Define a pattern of models"""

    # ID of the models
    wavenet_fft_cough = "80cb7d5b4ff7af169e74b3617c43580a41d5de5bd6c25e3251db2d11213755cd"
    srnn_cough = "cbba48a801f8b21600818da1362c61aa1287d81793e8cc154771d666956bdcef"

    # THE MODELS PATTERN defines which checkpoint (id, epoch) generates for how long (seconds)

    stream = Pseq([
        Pbind(
            "type", mmk.Checkpoint,
            "id", wavenet_fft_cough,
            "epoch", Prand([40, 50], inf),
            "seconds", Pwhite(lo=3., hi=5., repeats=1)
        ),
        Pbind(
            # This event inserts the most similar continuation from the Trainset "Cough"
            "type", mmk.NearestNextNeighbor,
            "soundbank", soundbank,
            "feature", mmk.Spectrogram(n_fft=2048, hop_length=512, coordinate="mag"),
            "seconds", Pwhite(lo=2., hi=5., repeats=1)
        ),
        Pbind(
            "type", mmk.Checkpoint,
            "id", srnn_cough,
            "epoch", Prand([200, 300], inf),
            # SampleRNN Checkpoints work best with a temperature parameter :
            "temperature", Pwhite(lo=.25, hi=1.5),
            "seconds", Pwhite(lo=.5, hi=2.5, repeats=1),
        )
    ], inf).asStream()

    """### Generate"""

    TOTAL_SECONDS = 30.

    ensemble = mmk.Ensemble(
        TOTAL_SECONDS, BASE_SR, stream,
        # with this you can print the event -- or not
        print_events=False
    )

    def process_outputs(outputs, bidx):
        for output in outputs[0]:
            mmk.audio(output.cpu().numpy(), sr=BASE_SR)

    loop = mmk.GenerateLoop(
        network=ensemble,
        dataloader=prompts,
        inputs=(h5m.Input(None,
                          getter=h5m.AsSlice(dim=1, shift=-BASE_SR, length=BASE_SR),
                          setter=h5m.Setter(dim=1)),),
        n_steps=int(BASE_SR * ensemble.max_seconds),
        add_blank=True,
        process_outputs=process_outputs
    )
    loop.run()