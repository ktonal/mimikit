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

- [ ] notebooks UI
    - [ ] Models with Config (NO NEW FEATURES! (io, etc...))
        - [ ] SampleRNN
        - [ ] FreqNet
        - [ ] S2S
        - [ ] remove HOMS
    - [ ] Factor Train ARM in TrainLoop
    - [ ] Cleanup GenerateLoop
        - [ ] **parameters
        - [ ] getters/setters
    - [ ] Feature Configs
        - [ ] MuLaw
        - [ ] FFT
    - [ ] File / Data View
    - [ ] Networks View
        - [ ] SampleRNN
        - [ ] FreqNet
        - [ ] S2S
    - [ ] Features View
    - [ ] Training View
    - [ ] Scripts
        - [ ] SampleRNN
        - [ ] FreqNet
        - [ ] S2S
    - [ ] define UI in notebooks
        - [ ] validate config
        - [ ] call main()
    - [ ] make_notebooks
        - [ ] hide code
        - [ ] embed UI state?
- [ ] Cleanup Callbacks and loggers
    - [x] CheckpointCallback
        - [x] remove h5 stuff
        - [x] save HP with OmegaConf
    - [ ] AudioLogger
        - [x] remove h5 stuff
        - [ ] pydub for audio write
- [ ] Upgrade pytorch lightning
- [ ] Ensemble NoteBook
- [ ] Output Evaluation NoteBook
- [ ] Clustering NoteBook
- [ ] Segmentation NoteBook
- [ ] UI Style sheet
- [ ] Cleanup Activations
- [ ] Mu/A Law
    - [ ] fix librosa mulaw
    - [ ] compression param
- [ ] Target Distributions
    - Scalar and Vector
        - [ ] Mixture of Logistics
        - [ ] Mixture of Gaussian 
        - [ ] Continuous Bernoulli (??) (--> correct VAE!!)
- [ ] New Features
    - [ ] Learnable STFT
    - [ ] MelSpec
    - [ ] MFCC
    - [ ] envelope(s)
    - [ ] clusters
    - [ ] segments
- [ ] Multiple Inputs
    - [ ] match network's inputs with features (& modules)
- [ ] Multiple Outputs
    - [ ] match network's output with targets (& features)
    - [ ] evaluate loss for each pair
    - [ ] layers can contribute to loss
    - [ ] GenerateLoop

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

 