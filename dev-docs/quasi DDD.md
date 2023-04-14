## Motivation

We want to
    
- easily compose ML models from *components* (inputs/outputs number and type, modules, network architecture, ...)   
- easily build UI interfaces to configure/instantiate them (and ideally, be able to save their state!)
    
### Top level structure

```
mimikit
- config
- view
- checkpoint
- io
    - audio
        - mu_law
        - spectrogram
        - enveloppe
        ...
    - labels
        - cluster_label
        - speaker_label
        ...
- information_retrieval
    - segment_audio
    - cluster
    ...
- modules
    - loss_fn
    ...
- networks
    - io_wrappers
    - sample_rnn
    - wavenet
    ...
- models
    - srnn
    - freqnet
    ...
- loops
    - train_loop
    - generate_loop
- trainings
    - train_arm
    - train_gan
    - train_diffusion
- scripts
    - generate_arm
    - train_freqnet
    - ensemble
    - eval_arm
    ....
- notebooks
    - generate_arm
    - train_freqnet
    - ensemble
    - explore_cluster
    .....
```

### ML Component Design Pattern

use `dataclasses` and inheritance to define & connect the layers of a 'ML component', e.g.

```
@dtc.dataclass
class NetConfig(Config):
    ...
    

@dtc.dataclass
class NetImpl(NetConfig, nn.Module):
    
    def __post_init__(self):
        # init modules...
    def forward(self, inputs, **kwargs):
        ...
    

@dtc.dataclass
class NetView(NetConfig, ConfigView):

    def __post_init__(self):
        # ... map params to widget ...
        ConfigView.__init__(self, **params)
```

Constructors are
- type-safe
- consistent across layers
- (de)serializable
- **defined once**
        
Then we can nest Configs, Impls & Views by doing:

```

@dtc.dataclass
class NestedConfig(Config):
    io: IOConfig
    net: NetConfig
    

@dtc.dataclass
class Model(NestedConfig, nn.Module):
    def __post_init__(self):
        ....
        
@dtc.dataclass
class ModelView(NestedConfig, ConfigView):
    ..... 
```

--> We win
- generic saving/loading Checkpoints
- highly expressive composition for io, models, features, views, etc...


### Implementation

different libraries / base classes offer different trade-offs between ease of use and ease of integrations with other libraries.

Ideally, we could,
- define a constructor -> `dataclass, attrs, namedtuple`
- have static type checker recognize it 
- attach it to a nn.Module -> inheritance?, `classmethod`?, decorator?
- use it in a View -> switch mutability
- (de)serialize it as config -> `OmegaConf`
- be able to export the nn.Module as `TorchScript`
