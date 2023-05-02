# Todos

## v0.4.4

### necessary

- SamplerTransformer
    - notebook
- integrate generate_chunks to GenerateLoop/Ensemble?

### nice to have

- TEST M1 Support
- Ensemble
    - Prompt UI
- Ensemble
    - working nearest neighbour generator
    - Generator classes (average from several models?, ...)
    - parameters (fade-in/out?, gain?, noise?, ...)
- Eval checkpoint Notebook
- Segmentation Notebook
- PocoNet / poco Wavenet
- Train notebook(s) with UI
- display waveforms with peaksjs and add metadata (prompt end, prompt position, temperature, event, ...)


### Experiment

- SamplerTransformer
    - faster generate (cached attention + single query)
    - Context Encoder (Encoder -> Decoder == context_seq -> generate_seq ~ hidden...)
- GMeans and batches
- TiedAE 
    - and batches
    - and residuals blocks
    - ==> SoundStream Logic (downsampling encoder)
- GMeans Layers
- TiedAE + GMeans
- SRNN, optimize first hidden in before_generate
- FreqNet and input/output Dropout when generating


### long term...

- support for TBPTT in freq domain
- SampleRNN in freq domain (tier_i ==> n_fft instead of frame_size)
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
- Hooks for
    - storing outputs
    - modifying generate_step() implementation on the fly...
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

 