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
        - [ ] S2S
            - [x] no HOM
            - [ ] from_config()
            - [ ] IOSpec
    - [ ] Factor Train ARM in TrainLoop
        - [ ] save configs
    - [ ] Cleanup GenerateLoop
        - [ ] Prompt Index
        - [ ] **parameters
        - [ ] getters/setters
        - [ ] move AudioLogger from Callback to Loop
- [ ] Views / UI
    - [ ] File / Data View
        - [x] FilePicker
        - [ ] FileUpload
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
        - [ ] Scripts
            - [ ] General Flow:
                IOSpec() -> soundbank
                Network(..., io_spec, ...) -> model
                Loop(soundbank, model, config)
            - [ ] SampleRNN
            - [ ] FreqNet
            - [ ] S2S
        - [ ] hide code
        - [ ] embed UI state?
        - [ ] test on colab
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
        - [ ] AR --> Input == Target --> shared data
        - [ ] Fixture --> no target --> data is read or passed
        - [ ] Auxiliary --> no input --> output is just collected
    - [ ] Batch Alignment for
        - [ ] Multiple SR
        - [ ] Multiple Domains
    - [ ] Same Variable, different repr (e.g. x_0 -> Raw, MuLaw --> ?)
    - [ ] Models support
    - [ ] TrainLoop Support
    - [ ] GenLoop Support
    - [ ] Logger/Display Support
- [ ] Target Distributions
    - Scalar and Vector
        - [ ] Mixture of Logistics
        - [ ] Mixture of Gaussian 
        - [ ] Continuous Bernoulli (??) (--> correct VAE!!)
    - [ ] Layers can contribute to loss
- [ ] New Features
    - [ ] KMer (seqPrior)
    - [ ] BitVector
    - [ ] TimeIndex
    - [ ] Learnable STFT
    - [ ] (Learnable) MelSpec
    - [ ] (Learnable) MFCC
    - [ ] Mixed Batch (fft + signal + clusters)
- [ ] New Networks
    - [ ] SampleGan (WaveGan with labeled segments?)
    - [ ] PocoNet
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

 