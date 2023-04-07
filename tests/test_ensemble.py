import h5mapper as h5m
import mimikit as mmk
from pbind import Pseq, Pbind, Prand, Pwhite, inf
import pytest
from assertpy import assert_that

from .test_utils import tmp_db


@pytest.fixture
def checkpoints(tmp_path):
    root = (tmp_path / "ckpts")
    root.mkdir()

    models = {
        'srnn': mmk.SampleRNN.from_config(mmk.SampleRNN.Config(
            io_spec=mmk.IOSpec.mulaw_io(mmk.IOSpec.MuLawIOConfig(sr=16000))
        )),
        'freqnet': mmk.WaveNet.from_config(mmk.WaveNet.Config(
            mmk.IOSpec.magspec_io(mmk.IOSpec.MagSpecIOConfig(sr=32000))
        ))
    }

    ckpts = {}
    for name, model in models.items():
        ckpt = mmk.Checkpoint(id=name, epoch=0, root_dir=str(root))
        ckpt.create(model)
        ckpts[name] = ckpt
    return ckpts


def test_should_generate(tmp_db, checkpoints):

    """# !! DON'T FORGET TO `%pip install pypbind` FIRST !!"""

    BASE_SR = 22050

    db = tmp_db("ensemble-test.h5")

    """### Define the prompts from which to generate"""

    prompts = next(iter(db.serve(
        (h5m.Input(data='signal', getter=h5m.AsSlice(shift=0, length=BASE_SR)),),
        shuffle=False,
        # batch_size=1 --> new stream for each prompt <> batch_size=8 --> one stream for 8 prompts :
        batch_size=3,
        sampler=mmk.IndicesSampler(
            # INDICES FOR THE PROMPTS :
            indices=(0, BASE_SR //2, BASE_SR)
        ))))[0]

    """### Define a pattern of models"""
    # THE MODELS PATTERN defines which checkpoint (id, epoch) generates for how long (seconds)

    stream = Pseq([
        Pbind(
            "generator", checkpoints["freqnet"],
            "seconds", Pwhite(lo=3., hi=5., repeats=1)
        ),
        # Pbind(
        #     # This event inserts the most similar continuation from the Trainset "Cough"
        #     "type", mmk.NearestNextNeighbor,
        #     "soundbank", soundbank,
        #     "feature", mmk.Spectrogram(n_fft=2048, hop_length=512, coordinate="mag"),
        #     "seconds", Pwhite(lo=2., hi=5., repeats=1)
        # ),
        Pbind(
            "generator", checkpoints["srnn"],
            # SampleRNN Checkpoints work best with a temperature parameter :
            "temperature", Pwhite(lo=.25, hi=1.5),
            "seconds", Pwhite(lo=.1, hi=1., repeats=1),
        )
    ], inf).asStream()

    """### Generate"""

    TOTAL_SECONDS = 10.

    ensemble = mmk.EnsembleGenerator(
        prompts, TOTAL_SECONDS, BASE_SR, stream,
        # with this you can print the event -- or not
        print_events=False
    )
    outputs = ensemble.run()

    assert_that(outputs.size(1)/BASE_SR).is_equal_to(TOTAL_SECONDS)