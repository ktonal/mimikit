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
            - [x] from_config()
            - [ ] IOSpec
        - [ ] update Checkpoint methods
    - [ ] Cleanup GenerateLoop
        - [x] Prompt Index
        - [x] **parameters
        - [x] getters/setters
        - [x] move AudioLogger from Callback to Loop
        - [x] migrate train loop 
        - [ ] migrate ensemble
- [ ] Views / UI
    - [ ] File / Data View
        - [x] FilePicker
        - [x] FileUpload
        - [ ] NB Data cell with colab support
    - [ ] IOSpec View
        - [ ] Features
        - [ ] IOFactory
    - [ ] Networks View
        - [x] SampleRNN
        - [x] WaveNet
        - [ ] S2S
    - [x] Training View
    - [ ] `Run` Button
    - [ ] define UI in notebooks
        - [ ] validate config
        - [ ] call main()
    - [ ] UI Style sheet
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
- [ ] Segmentation NoteBook
- [ ] Output Evaluation NoteBook
- [ ] New Features API:
    - [ ] unary callables
        - [ ] class ClusterLabel(Feature):
        - [ ] class SegmentLabel(Feature):
            - [ ] from rec mat
        - [ ] KMer (seqPrior)
        - [ ] BitVector
        - [ ] MelSpec
        - [ ] MFCC
        - [ ] TimeIndex
        - [ ] Frame
        - [ ] Cast
        - [ ] SignalToFile
        - [ ] PlotSeries
        - [ ] PlotSpectrogram
    - [ ] UI
    - [ ] Nice to have:
        - [ ] Scaler
            - [ ] MinMax
            - [ ] Normal
        - [ ] Augmentation(functional, prob)
        - [ ] Learnable STFT
        - [ ] (Learnable) MelSpec
        - [ ] (Learnable) MFCC
        - [ ] Mixed Batch (fft + signal + clusters)
        - [ ] Parametrized (one class, several params, e.g. q_levels=(2, 4, 8, ...))
- [ ] Multiple IO
    - [ ] tuple or not tuple
    - [ ] AR Feature vs. Fixture vs. Auxiliary Target (vs. kwargs)
        - [ ] AR --> Input == Target --> shared data
            prompt must be: prior_t data + n_steps blank
            !! target interface must come from data
        - [ ] Fixture --> no target 
            --> data is read
            prompt must be: prior_t + n_steps data
            !! this modifies the length of the Dataset!
            --> data is transformed from (possibly AR) input
            !! this DOESN'T modify the length
        - [ ] Auxiliary --> no input --> output is just collected
            prompt must be: priot_t + n_steps blank
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
        - [x] Mixture of Logistics
        - [x] Mixture of Gaussian 
        - [ ] Continuous Bernoulli (??) (--> correct VAE!!)
        - [ ] WaveRNN style SoftMax
    - [ ] Layers can contribute to loss (e.g. ELBO)
- [ ] New Networks
    - [ ] SampleGan (WaveGan with labeled segments?)
    - [ ] PocoNet
    - [ ] Stable Diffusion Experiment
- [ ] Test utils
    - [ ] test soundbank
    - [ ] test model
    - [ ] test checkpoint
    
    
#### Future nice-to-have
- [ ] DataTransforms / Feature.compose
    - [ ] MinMaxScaler
    - [ ] StandardScaler
    - etc....
- [ ] Loss Terms
- [ ] Hooks for storing outputs
- [ ] flowtorch
- [ ] Multi-Checkpoint Models (stochastic averaging)
- [ ] Resampler classes with `n_layers`
- [ ] jitability
- [ ] Network Visualizer (UI)
- [ ] Resume Training
    - [ ] Optimizer in Checkpoint
- [ ] Upgrade python 3.9 ? (colab is 3.7.15...)
- [ ] stacking of models in Ensemble
- [ ] M1 Support

 