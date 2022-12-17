from generator import get_encoder, get_decoder, VAE, kl_anneal
from utils import chopping, padding, onehot_encoding, onehot_decoding
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser(description="Parameters for training the VAE",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data", type=str,
                    help="filepath to the training data csv file, should have a sequence column and no index")
parser.add_argument("-m", "--max_seq", default=48, type=int,
                    help="Maximum sequence length")
parser.add_argument("-l", "--latent", default=50, type=int,
                    help="Dimension of the latent space")
parser.add_argument("-e", "--epochs", default=2, type=int,
                    help="Number of training epochs")
parser.add_argument("-b", "--batch", default=50, type=int,
                    help="Batch size")
parser.add_argument("-a", "--anneal", default='logistic', type=str,
                    help="KL term annealing function either normal, linear or logistic")
parser.add_argument("-k", "--k", default=0.1, type=float,
                    help="k term in logstic function")
parser.add_argument("-x", "--x0", default=1, type=int,
                    help="x0 term in logistic function")
parser.add_argument("--name", default='last', type=str,
                    help="file name to save the trained model")

args = parser.parse_args()
config = vars(args)

data_path = config['data']
max_seq_length = config['max_seq']
latent_dim = config['latent']
epochs = config['epochs']
batch_size = config['batch']
anneal_type = config['anneal']
k = config['k']
x0 = config['x0']
name = config['name']

data = pd.read_csv(data_path)
print('##########\n')
print(data.head())
train_data = np.array(onehot_encoding(
    padding(chopping(
        data['sequence'].tolist(), lim=max_seq_length), lim=max_seq_length)), dtype='float64')

encoder = get_encoder(latent_dim=latent_dim)
decoder = get_decoder(latent_dim=latent_dim)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))

print('##########\n')
losses = []
recon_losses = []
kl_losses = []
kl_weights = []


for epoch in range(epochs):
    if anneal_type == 'normal':
        pass
    else:
        new_val = kl_anneal(epoch, anneal_type, k, x0)
        tf.keras.backend.set_value(vae.beta_var, new_val)
    print(f'epoch number: {epoch}, current beta: {vae.beta_var}')
    history = vae.fit(train_data, epochs=1, batch_size=batch_size)
    losses.append(history.history['loss'][0])
    recon_losses.append(history.history['reconstruction_loss'][0])
    kl_losses.append(history.history['kl_loss'][0])
    kl_weights.append(history.history['kl_weight'][0])

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes[0, 0].plot(losses,)
axes[0, 0].set_title('Total Loss')
axes[0, 1].plot(recon_losses,)
axes[0, 1].set_title('Reconstruction Loss')
axes[1, 0].plot(kl_losses,)
axes[1, 0].set_title('KL Loss')
axes[1, 1].plot(kl_weights,)
axes[1, 1].set_title('KL Weight')

axes[1, 0].set_xlabel('epoch')
axes[1, 1].set_xlabel('epoch')
plt.tight_layout()
fig.savefig('../tmp/losses.png')

encoder.save('../saved_models/user/generator/encoder'+name)
decoder.save('../saved_models/user/generator/decoder'+name)

# TODO save trained model in directory
