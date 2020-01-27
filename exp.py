#!/usr/bin/env python
# coding: utf-8

import pdb
import logging
import pickle
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.datasets as data
import numpy as np
import pandas as pd

from datetime import datetime
from sacred import Experiment
ex = Experiment('test')

out_path = './output'

# Create a logger
logger = logging.getLogger('test')

# Add a logging file
logpath = '{}/logs'.format(out_path)
logtime = datetime.now().strftime('%Y-%m-%d-%H%M:%S:%f')
fh = logging.FileHandler('{}/{}.log'.format(logpath, logtime))
fh.setLevel(logging.INFO)

formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Tie this to the experiment
ex.logger = logger

from sacred.observers import TinyDbObserver
ex.observers.append(TinyDbObserver('test'))

'''
This is the configuration scope, which is a function executed by sacred just
before running the experiment, with all defined variables saved into our
database as configuration parameters

For more information, see
https://sacred.readthedocs.io/en/stable/configuration.html
'''
@ex.config
def my_config():
    np_seed=1

    # Data generating parameters
    n_train_samp=1000
    n_test_samp=1000
    n_features=1000
    sparsity=0.1

    # Because this is a function, we can dynamically set other config params
    n_informative = int(n_features * sparsity)

    # Model Parameters for logistic regression
    alpha=1.

'''
This is a captured function, which means that the arguments are automatically
injected from our configuration.  In our main script, decorated with
@ex.automain, this will allow us to just run it without any arguments, to keep
things simple
'''
@ex.capture
def get_data(np_seed, n_features, n_train_samp, n_test_samp, n_informative):
    return data.make_regression(
            n_samples=n_train_samp + n_test_samp,
            n_features=n_features,
            n_informative=n_informative,
            coef=True,
            random_state=np_seed)


@ex.capture
def get_model(alpha):
    return lm.Lasso(alpha)

'''
This is the main script that will be run for our experiment.  Note that we use
ex.info[] to capture custom metrics, and ex.add_artifact to link any files
that are produced during the experiment, so we can easily refer to them later
'''
@ex.automain
def run(np_seed, n_train_samp):
    np.random.seed(np_seed)

    X, y, coef = get_data()

    X_train, X_test, y_train, y_test = \
        ms.train_test_split(X, y, train_size=n_train_samp, random_state=np_seed)

    clf = get_model()
    clf.fit(X_train, y_train)

    ex.info['rmse_y'] = np.sqrt(np.mean((y_test - clf.predict(X_test))**2))
    ex.info['rmse_coef'] = np.sqrt(np.mean((coef - clf.coef_)**2))

    time_str = datetime.now().strftime('%Y-%m-%d-%H%M:%S:%f')
    file_str = '{}/artifacts/{}-model.pkl'.format(out_path, time_str)

    with open(file_str, 'wb') as f:
        pickle.dump(clf, f)

    ex.add_artifact(file_str, 'model')

    # We don't really use this, because we already record in info
    return clf.score(X_test, y_test)
