from code.utils import chopping, padding, onehot_encoding, onehot_decoding
from code.utils import prepare_CNN, prepare_VAE, point_prep, point_to_mic
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#encoder = keras.models.load_models()
decoder = keras.models.load_model(
    '../saved_models/paper/generator/VAE_v2_decoder')
regressor_cnn = keras.models.load_model(
    '../saved_models/paper/regressor/CNN_gr_neg')

print(decoder.summary())
print(regressor_cnn.summary())

num_random = 500
num_select = 10

zpoint = np.zeros((1, 50))
novo_peptides = []
novo_peptides_cnn_mic = []
novo_peptides_rnn_mic = []

for i in range(num_random):
    rpoint = np.random.default_rng().uniform(-5, 5, 50)
    pep = onehot_decoding(
        decoder(zpoint+rpoint).numpy().tolist())[0].strip('-')
    if ('-' not in pep) & (len(pep) > 35):
        novo_peptides.append(pep)
        novo_peptides_cnn_mic.append(regressor_cnn(
            point_prep(zpoint+rpoint)).numpy()[0][0])
        # novo_peptides_rnn_mic.append(regressor_rnn(point_prep(zpoint+rpoint)).numpy()[0][0])

df = pd.DataFrame({"sequence": novo_peptides,
                   "cnn_mic": novo_peptides_cnn_mic, })
# 'rnn_mic': novo_peptides_rnn_mic})

print(df)
