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

## Design

```python
class Entity(Protocol):

    def save(self, **python_objects) -> Identifiers:
        ...

    def load(self, **identifiers) -> PythonObject:
        ...

class Config(Entity[Union[str, Path], Dataclass]):
    ...

class Audio(Entity[Union[str, Path], np.ndarray]):
    pass

class Model(Entity[Union[str, Path], nn.Module]):
    pass

class Repo[Id, Entity]:
    pass

class Feature[Entity, Optional[Tuple[Callable, ...]]]:
    pass
```

## Todo

- [ ] Upgrade python 3.9 ? (colab is 3.7.15...)
- [ ] Upgrade pytorch lighnting
- [ ] Use Protocols for feature/models
- [ ] notebooks UI 
    - [x] define TrainARMConfig
    - [x] cleanup train
    - [x] define networks configs
    - [x] define scripts mains
    - [ ] define UI in notebooks -> call mains!  
- [ ] Ensemble Demo
- [ ] Output Evaluation Demo
- [ ] Clustering Demo
- [ ] Segmentation Demo
- [ ] stacking of models in Ensemble
 