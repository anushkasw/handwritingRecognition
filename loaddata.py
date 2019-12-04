#! /bin/env python3

import pickle
import numpy as np


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


data = load_pkl('train_data.pkl')
np.save('finalDataTrain.npy', data)
