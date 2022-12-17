from code.utils import chopping, padding, onehot_encoding, onehot_decoding
from code.utils import prepare_CNN, prepare_VAE
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


decoder = keras.models.load_model(
    'saved_models/paper/generator/VAE_v2_decoder')
regressor = keras.models.load_model(
    'saved_models/paper/regressor/CNN_gr_neg')


def point_prep(point):
    return prepare_CNN(onehot_decoding(decoder(point).numpy().tolist()))


zpoint = np.zeros((1, 50))
novo_peptides = []
novo_peptides_mic = []

num_random = 500
for i in range(num_random):
    rpoint = np.random.default_rng().uniform(-5, 5, 50)
    pep = onehot_decoding(
        decoder(zpoint+rpoint).numpy().tolist())[0].strip('-')
    if ('-' not in pep) & (len(pep) > 35):
        novo_peptides.append(pep)
        novo_peptides_mic.append(regressor(
            point_prep(zpoint+rpoint)).numpy()[0][0])

df = pd.DataFrame({"sequence": novo_peptides,
                   "mic_pred": novo_peptides_mic, })

df.to_csv('outfile.csv')
