# mimikit

The MusIc ModelIng toolKIT (`mimikit`) is a python package that does Machine Learning with audio data.

Currently, it focuses on 
- training auto-regressive neural networks to generate audio 

but it does also contain code to perform
- basic & experimental clustering of audio data  
- segmentation of audio files
 

## Usage 

Head straight to the [notebooks](https://github.com/ktonal/mimikit-notebooks) for example usage of `mimikit`, or open them directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ktonal/mimikit-notebooks/blob/main)

## Output Samples

You can explore the outputs of different trainings done with `mimikit` at this demo website:

   https://ktonal.github.io/mimikit-demo-outputs 

## License

`mimikit` is distributed under the terms of the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)


## Todo

#### v0.4.0

- [ ] code cleanup
    - [ ] Models with Config (NO NEW FEATURES! (io, etc...))
        - [x] SampleRNN
            - [x] no HOM
            - [x] from_config()
            - [x] IOSpec
            - [x] loss fn
        - [x] WaveNet
            - [x] no HOM
            - [x] from_config()
            - [x] IOSpec
            - [x] loss fn
        - [ ] S2S
            - [x] no HOM
            - [ ] from_config()
            - [ ] IOSpec
        - [x] remove HOMS module and impls
    - [X] Factor Train ARM in TrainLoop
        - [ ] save configs
    - [ ] Cleanup GenerateLoop
        - [ ] **parameters
        - [ ] getters/setters
        - [x] move AudioLogger from Callback to Loop
    - [x] Cleanup Callbacks and loggers
        - [x] CheckpointCallback
            - [x] remove h5 stuff
            - [x] save HP with OmegaConf
        - [x] AudioLogger
            - [x] remove h5 stuff
            - [x] pydub for audio write
    - [X] Feature Configs
        - [X] MuLaw
        - [X] FFT
        - [x] batch items units in samples
    - [x] Cleanup Loss Functions
    - [ ] Cleanup Activations
    - [ ] IOSpec
        - [ ] layers can contribute to loss
        - [ ] GenerateLoop
    - [ ] Scripts
        - [ ] General Flow:
            IOSpec() -> soundbank
            Network(..., io_spec, ...) -> model
            Loop(soundbank, model, config)
        - [ ] SampleRNN
        - [ ] FreqNet
        - [ ] S2S
- [ ] Views / UI
    - [ ] File / Data View
    - [ ] Networks View
        - [ ] SampleRNN
        - [ ] FreqNet
        - [ ] S2S
    - [ ] Features View
    - [ ] Training View
    - [ ] `Run` Button
    - [ ] define UI in notebooks
        - [ ] validate config
        - [ ] call main()
    - [ ] make_notebooks
        - [ ] hide code
        - [ ] embed UI state?
        - [ ] test on colab
- [X] Upgrade pytorch lightning
- [ ] Ensemble NoteBook
- [ ] Clustering NoteBook
    - [ ] class ClusterBank(h5m.TypedFile):
    - [ ] class ClusterLabel(Feature):
- [ ] Segmentation NoteBook
    - [ ] class Envelope(Feature):
    - [ ] class SegmentLabel(Feature):
- [ ] Output Evaluation NoteBook
- [ ] UI Style sheet
- [ ] Mu/A Law
    - [ ] fix librosa mulaw
    - [ ] compression param for MuLaw
    - [ ] FFT window
- [ ] class DataBank
    - [ ] Definition, Creation
    - [ ] integration with Features
- [ ] Multiple IO
    - [ ] AR Feature vs. Fixture vs. Auxiliary Target (vs. kwargs)
    - [ ] Models support
    - [ ] TrainLoop Support
    - [ ] GenLoop Support
- [ ] Target Distributions
    - Scalar and Vector
        - [ ] Mixture of Logistics
        - [ ] Mixture of Gaussian 
        - [ ] Continuous Bernoulli (??) (--> correct VAE!!)
- [ ] New Features
    - [ ] KMer (seqPrior)
    - [ ] BitVector
    - [ ] TimeIndex
    - [ ] Learnable STFT
    - [ ] (Learnable) MelSpec
    - [ ] (Learnable) MFCC
    - [ ] Mixed Batch (fft + signal + clusters)
- [ ] New Networks
    - [ ] SampleGan (WaveGan with labeled segments)
    - [ ] Stable Diffusion Experiment
    
    
#### Future nice-to-have

- [ ] Hooks for storing outputs
- [ ] Multi-Checkpoint Models (stochastic averaging)
- [ ] Resampler classes with `n_layers`
- [ ] Network Visualizer (UI)
- [ ] Resume Training
    - [ ] Optimizer in Checkpoint
- [ ] Upgrade python 3.9 ? (colab is 3.7.15...)
- [ ] stacking of models in Ensemble
- [ ] M1 Support

 