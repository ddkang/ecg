from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import os
import time

import load
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import keras

VERBOSE = False

def count_inversions(preds, inv_len=5):
    def inverted(seq):
        if seq[0] == seq[-1]:
            return len(set(seq)) > 1
        else:
            return False

    nb_inv = 0
    for ind in range(inv_len + 2, len(preds)):
        if inverted(preds[ind - inv_len - 2 : ind]):
            nb_inv += 1
    return nb_inv

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict(args.data_json, args.model_path)
