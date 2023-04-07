# Todos

## v0.4.0

### necessary

- TEST EnsembleGenerator
- Seq2Seq
- Notebooks
    - FreqNet
    - SampleRNN
    - Seq2Seq
    - Generate from Checkpoint
    - Ensemble
- packaging
    - publish PeaksJSWidget
    - publish pbind
    - new build tools...?
    - dependencies
- Colab test

### nice to have

- TEST M1 Support
- Ensemble / Generate nb
    - Prompt UI
    - AudioLogger methods (batch, spectrogram...)
- Ensemble
    - working nearest neighbour generator
    - Ensemble classes (average from several models?, fade-in/out?)
- Clusterizer App
- Eval checkpoint Notebook
- Segmentation Notebook
- PocoNet / poco Wavenet

    
### long term...

- flexible IO declaration (fft, signal+segments, learnable fft, ...)
- more audio features
    - class ClusterLabel(Feature):
    - class SegmentLabel(Feature):
        - from rec mat
    - KMer (seqPrior)
    - BitVector
    - Quantize / Digitize / Linearize
    - [/] MelSpec
    - [/] MFCC
    - TimeIndex
    - Scaler
        - MinMax
        - Normal
    - Augmentation(functional, prob)
    ...................................
    - tuple or not tuple
    - AR Feature vs. Fixture vs. Auxiliary Target (vs. kwargs)
        - AR --> Input == Target --> shared data
            prompt must be: prior_t data + n_steps blank
            !! target interface must come from data
        - Fixture --> no target 
            --> data is read
            prompt must be: prior_t + n_steps data
            !! this modifies the length of the Dataset!
            --> data is transformed from (possibly AR) input
            !! this DOESN'T modify the length
        - Auxiliary --> no input --> output is just collected
            prompt must be: priot_t + n_steps blank
    - Batch Alignment for
        - Multiple SR
        - Multiple Domains
    - Same Variable, different repr (e.g. x_0 -> Raw, MuLaw --> ?)

- More Networks
    - SampleGan (WaveGan with labeled segments?)
    - Stable Diffusion Experiment
    - FFT_RNN
    ....
- Loss Terms
- Hooks for storing outputs
- flowtorch
- huggingface/dffusers/transformers
- Multi-Checkpoint Models (stochastic averaging)
- Resampler classes with `n_layers`
- jitability / torch==2.0 compile()
    - no `*tuple` expr...
- Network Visualizer (UI)
- Resume Training
    - Optimizer in Checkpoint
- Upgrade python 3.9 ? (colab is 3.7.15...)

 