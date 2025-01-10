# Todos

## v0.4.4

### necessary

- [x] SamplerTransformer
    - [ ] notebook
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
- [ ] input/output grabber

### Experiment

- textbooks
    - understand (better!)
        - [ ] ODE basics
        - [ ] Lattice Filter section
    - summarize (math Cheat sheet)
        - Linear Prediction of Speech
    - read 
        - more markel
        - more Chu 
        - more DSP  
    - implement
        - [ ] Chu period estimation (vs Markel period estimation)
        - [ ] Markel 
            - [ ] basics
            - [ ] chapter 2 (Formulations)
- pitch detection
    - single
        - check sigmund?
        - 
    - multi
        - check melodyne patent
        - check pitch tracking from librosa

- stretching
    - segment vocoder 
        - phase advance as a cycle over k frames?
        - fft of the phase signals?

- fiddle with
    - zero padded FFT from / to zero crossings
    - inv_fourrier(F0 filter) -> autocorrelation
    - eigenvectors of the autocorrelation of S[0:T] (=> denoising? )
    - (non) stationarity of (some formulation of) the autocorrelation function/matrix -> segmentation
        - autocorrelation of the autocorrelation 
    - analytic signal 
        - of (cos(a) + cos(b))
        - low pass filtering of instantaneous freq / amplitude
    - cascaded long term LPCs (of autocorrelation peaks)
    - filter + attenuation (Chu p. 317-321 -> enhanced periodicity)
    - wave modulation
    - fractional period estimation
    - prony's method and similar
    - bispectrum
    - karhunen-loeve transform
    - exact estimation of z (and its harmonics?)
        - fitting of a comb filter?
    - onset detection with matched filter [1, 1, 1, 1, -1, -1, -1, -1]
    
- harmonicity / inharmonicity
    - local covariance of S vs. Gaussian noise
    
        
- autocorrelation
    - stretch (ac_normed * gain(S))
    - transient / harmonicity detector from hpss -> ar_harm, ar_perc
    - deconvolve(yt, ac(yt)) --> ???
    - dct(ac(yt)) for pitch tracking
    - levinson durbin coeff
    - find zi that maximizes Y(z) for a given y(t) (through gradient descent?)
        (init zi on unit circle with auto-correlation peaks, and optimize for the angle(z))

- overfiter
    - [x] dilated layers  --> X
    - [x] growing k layers (5, 4, 3, 2)
        - [x] taking input  --> X
        - [x] sequential  --> X
    - [ ] attention like mixing of inputs
    - [x] PBits targets
    - [x] {sign(tanh) *} softplus
    - [x] pred as diff to last sample
    - [ ] discretized logistic targets
    - [ ] seq2seq
    - [ ] MoE
    - [ ] time / input embedding
    - [ ] resampled / filtered inputs
    
- Context Encoder -> autoregressive net
- audio Unet
    - [ ] stable diff?
    - [ ] seq2seq
     
- BinRNN (Sample Rnn with ffts)
- GMeans and batches
- [ ] VAEs
    - TiedAE 
        - and batches
        - and residuals blocks
        - ==> SoundStream Logic (downsampling encoder)
        - [ ] encoding as FreqNet input
        - [ ] encoding as FreqNet condition
- GMeans Layers
- TiedAE + GMeans
- FreqNet
    - [ ] input/output Dropout/Fadeout when generating
    - [ ] with DCT
- FFT phase estimation
    - through weighted loss function (weight = magnitude)
    - through Max likelihood

### long term...

- support for TBPTT in freq domain
- SampleRNN in freq domain (tier_i ==> n_fft instead of frame_size)
- flexible IO declaration (fft, signal+segments, learnable fft, ...)
- more audio features
    - class ClusterLabel(Feature):
    - class SegmentLabel(Feature):
        - from rec mat
    - KMer (seqPrior)
    - [x] BitVector
    - Quantize / Digitize / Linearize
    - [/] MelSpec
    - [/] MFCC
    - TimeIndex
    - Scaler
        - MinMax
        - Normal
    - Augmentation(functional, prob) with audiomentation?
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
    - LPCNet (or similar fitting of residuals)
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
- [x] Resume Training
    - Optimizer in Checkpoint
- Upgrade python 3.9 ? (colab is 3.7.15...)

 