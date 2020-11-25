# Music Modelling Toolkit (MMK)

MMK is a library of generative models for audio and music. 
It is fast, free, expressive and lets you use your own data,
which makes it perfect for individuals with low-resources and little Machine-Learning/Programming expertise
who wish to get their hands on those technologies in their own terms, e.g. artists, sound-designers, developers...      

The `notebooks/` can be used in google colab to train models and generate sounds with them. Try it out and have fun!

## Quickstart

#### Setup

- First, make sure you have a [Google Account](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp)

- If you don't have `conda` installed, install it from the [official website](https://www.anaconda.com/products/individual)
or - if you're not going to use `python` much - install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
which has everything `mmk` needs and is only 400Mb instead of 5Gb for the full anaconda version...

- Then, in order to store, share and keep track of your models, we recommend that you create yourself an account on [Neptune](https://neptune.ai/).
This is a great service and the free version comes with 100Gb for free!

- Now follow these [instructions](https://docs.neptune.ai/security-and-privacy/api-tokens/how-to-find-and-set-neptune-api-token.html#how-to-setup-api-token) 
to setup your api_token on your personal computer.

- Log in to your neptune account and create a new project where you will store your data and an other one where you will store your models.  

- Next, open a terminal and paste the following lines to install `mmk` and its dependencies to your home directory :

```
git clone https://github.com/antoinedaurat/mmk.git
pip install -r mmk/requirements.txt
```

#### Upload Data

- Then, still in the terminal, open a jupyter notebook by typing

```
jupyter notebook
```

- In the jupyter notebook, open `mmk/notebooks/Make and Upload Database.ipynb` and follow the instructions.

- Once you successfully uploaded some data to your neptune account 
    - (save and) close the jupyter notebook
    - go back to your terminal
    - stop the process by pressing Ctrl-C

#### Train a model

- Go to [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb]) 

- In the `File` menu, choose `upload notebook` and select `mmk/notebooks/FreqNet.ipynb` to be uploaded.

- configure the code and run the cells 

That's it! Happy Deep-Learning! 
 

## Models

There is currently one available model : `FreqNet`. Check out `notebooks/freqnet.ipynb` for its usage and documentation.