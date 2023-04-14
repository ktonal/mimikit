# mimikit

The MusIc ModelIng toolKIT (`mimikit`) is a python package that does Machine Learning with audio data.

Currently, it focuses on training auto-regressive neural networks to generate audio.

but it does also contain an app to perform basic & experimental clustering of audio data in a notebook.

## Installation

you can install with pip
```shell script
pip install mimikit[torch]
```
or with 
```shell script
pip install --upgrade mimikit[torch]
```
if you are looking for the latest version
 
for an editable install, you'll need
```shell script
pip install -e . --config-settings editable_mode=compat
```

## Usage 

Head straight to the [notebooks](https://github.com/ktonal/mimikit-notebooks) for example usage of `mimikit`, or open them directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ktonal/mimikit-notebooks/blob/main)

## Output Samples

You can explore the outputs of different trainings done with `mimikit` at this demo website:

   https://ktonal.github.io/mimikit-demo-outputs 

## License

`mimikit` is distributed under the terms of the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)
