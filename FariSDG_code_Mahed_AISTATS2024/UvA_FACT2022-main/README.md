# Experiment 1

This code was derived from the replication study of Wang et al 2022, and as
such much of the code matches there implementation. The original README for
this package is below. 

We have made a very small number of changes to this study for our needs. Namely, 
we have increased the number of iterations used in the MLP classifer, and we
have added a logistic regression. Additionally we have add our models. 

It is important to note, when using this code that the replication code will
save models, and therefore to compare between different splits it is important to 
clear the saved models. This can be done in several ways, either by running the
replicates on separate machine, or by running one at a time and removing the
cache, or by changing the cache directory for each run "train.models_dir".

For simplicity we have simply provided the general script (without
our rep splitting), but please bear this in mind when running the models.



# Testing the Replicatability of DECAF



The original code of DECAF paper is [here]( https://github.com/vanderschaarlab/DECAF)

## Prerequisites

Code is compatible with Python version 3.8.* due to explicit version requirements of some of the project dependencies.

We use [DVC](https://dvc.org/) for storing trained moodels.

## Installation

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Downloading pretrained models

If you want to run the notebooks with pretrained models you need to download them first by running:

```
dvc pull
```

## Contents

`train.py` - Training functions for different GAN models, including DECAF, Vanilla_GAN, WGAN_gp, FairGAN.

`data.py` - Loading data and preprocessing functions for Adult dataset and Credit dataset.

`metrics.py` - Measuring the Data Quality and Fairness.

`experiment_1.ipynb`, `experiment_2.ipynb ` - A quick overview of reproduction results.

`run_experiment_1.py`, `show_results.py` - Scripts used for generating final results for Experiment 1.
