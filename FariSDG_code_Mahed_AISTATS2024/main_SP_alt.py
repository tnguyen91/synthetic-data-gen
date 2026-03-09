from spgan import train_spgan
import numpy as np
import data_loading
import utils
import argparse
import copy
#import pdb
#pdb.set_trace()


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=40, help="dimensionality of the latent space")
parser.add_argument("--lam", type=float, default=0.01, help="regularisation coef")
parser.add_argument("--n_critic", type=int, default=20, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

X_in = data_loading.Data_Loading_Adult(1)
X_gen = train_spgan(X_in, opt)
