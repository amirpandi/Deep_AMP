from regressor import rnn_regressor, cnn_regressor
from utils import chopping, padding, onehot_encoding, onehot_decoding
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser(description="Parameters for training the VAE",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_pos", type=str,
                    help="filepath to the AMP training data csv file, should have a sequence column, a value column and no index")
parser.add_argument("-n", "--data_neg", type=str,
                    help="filepath to the non-AMP training data csv file, should have a sequence column and no index")
parser.add_argument("-a", "--model_architecture", default='cnn', type=str,
                    help="Regressor model architecture, either rnn or cnn")
parser.add_argument("-m", "--max_seq", default=48, type=int,
                    help="Maximum sequence length")
parser.add_argument("-e", "--epochs", default=2, type=int,
                    help="Number of training epochs")
parser.add_argument("-b", "--batch", default=50, type=int,
                    help="Batch size")
parser.add_argument("--name", default='last', type=str,
                    help="file name to save the trained model")

args = parser.parse_args()
config = vars(args)

model_architecture = config['model_architecture']
data_pos_path = config['data_pos']
data_neg_path = config['data_neg']
max_seq_length = config['max_seq']
epochs = config['epochs']
batch_size = config['batch']
name = config['name']

data_pos = pd.read_csv(data_pos_path)
data_neg = pd.read_csv(data_neg_path)


if model_architecture == 'cnn':
    model = cnn_regressor()
elif model_architecture == 'rnn':
    model = rnn_regressor()
else:
    raise ValueError('Model should be either rnn or cnn')
