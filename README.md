# Setting up experiments with `sacred`

## Overview 

`sacred` [link](https://github.com/IDSIA/sacred) is a python package that exists to help run and manage experiments in a reproducible way.  This is a minimal working example of an experiment set up to run with `sacred`, which is meant to serve as a template for quickly running other ML experiments.

Generally speaking, I have found this to be a good tool for experiments where you want to:
* Run an algorithm and record several custom metrics
* Do this multiple times over a range of hyperparameters
* Run the experiment asynchronously from analyzing the results
* Retain the ability to drill-down on individual runs (e.g., loading a trained model to inspect)

While `sacred` supports using MongoDB + a few browser-based interfaces to analyze results, I have found these to be too clunky for my use-case, and use the combination of `tinydb` and Jupter Notebooks instead.

## Toy Experiment

In this toy example, we run the following experiment:
* Generate synthetic data of a sparse linear regression task
* Fit a linear model using `l1` regularization
* Record held-out RMSE for prediction and parameter recovery

We vary the number of training samples to construct a learning curve, seeing how our test RMSE goes down as our training sample size goes up.  Of course, in a real experiment, we might want to tune the regularization parameter for each value of `n_train_samp`, but we keep it constant here for simplicity.  For each setting of `n_train_samp`, we run 10 independent experiments, generating new data each time.

NOTE: We overload the word "experiment" to mean both (a) each individual run, and (b) the entire collection of runs.

## Walk-through

The main files used here are as follows:
* `run.sh` is a bash script that loops over experiment parameters and calls `exp.py` for each individual experiment run
* `exp.py` contains the logic for each individual run, demonstrating the use of `sacred` to
  + Auomatically log hyperparameters using the `@ex.config` decorator
  + Record custom metrics using the `ex.info` dictionary
  + Link to saved files (in this case, saved models using `pickle`), using `ex.add_artifact`
* `analyze_results.ipynb` is a short notebook that loads the experimental results and demonstrates how to analyze them.

All experimental results are stored in `./output`, which includes
* `logs/` for logging files (demonstrating integration with the python `logging` package)
* `results/` for the `tinydb` storage of experimental results
* `artifacts/` for the saved files created during our experiment (e.g., picked models)

I also include a `clean_results.sh` script for deleting previous experimental results.

## Requirements

In addition to the dependencies required for this particular experiment (e.g., `numpy`, `pandas`, `sklearn`, etc), this setup requires the installation of `sacred` and a couple of `tinydb` dependencies, which you can install using 
```
pip install sacred
pip install hashfs
pip install tinydb
pip install tinydb-serialization
```
