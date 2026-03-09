import argparse
from tqdm import tqdm
import os
import numpy as np
import copy
import math
import sys
import pandas as pd

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, 15),
            #nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, z):
        # gen_input = torch.cat((a, z), -1)
        sam = self.model(z)
        # img = img.view(img.shape[0], *img_shape)
        return sam


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(15, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # img_flat = img.view(img.shape[0], -1)
        validity = self.model(x)
        return validity


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(14, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 20),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x)
        y = y.clamp(min=1e-4,max=1-1e-4)
        return y


def train_ifgan(X_in, opt, storeStats=False):
    if storeStats:
        stats_store = []


    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator()
    dec1 = Decoder()
    dec2 = Decoder()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader

    X = torch.Tensor(X_in)

    D = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(D, batch_size=opt.batch_size, shuffle=True, )

    loss_dec = torch.nn.BCELoss(reduction="none")

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_dec1 = torch.optim.Adam(dec1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_dec2 = torch.optim.Adam(dec2.parameters(), lr=opt.lr, betas=(0.5, 0.999))


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in tqdm(range(opt.n_epochs)):

        batches_done = 0

        for i, x in enumerate(dataloader):

            # Configure input
            # real_sample = Variable(x.type(Tensor))
            real_sample = x[0]
            # sen_att = x[1]
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            optimizer_dec1.zero_grad()
            optimizer_dec2.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (len(x[0]), opt.latent_dim))))

            # Generate a batch of images

            fake_sample = generator(z).detach()
            fake_wo_sen = fake_sample.clone()
            fake_wo_sen[:, 9:10] = 0
            y_fake = fake_sample[:, -1]

#            y_fake = y_fake >0.5

            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_sample)) + torch.mean(discriminator(fake_sample))

            dec1_out = dec1(fake_sample[:, :-1]).float()
            loss_d1t = loss_dec(dec1_out, y_fake.reshape(-1,1).float())

            dec2_out = dec2(fake_wo_sen[:, :-1]).float()
            loss_d2t = loss_dec(dec2_out, y_fake.reshape(-1,1).float())


            loss_d1 = torch.mean(loss_d1t)
            loss_d2 = torch.mean(loss_d2t)

            loss_D.backward()
            optimizer_D.step()

            loss_d1.backward()
            optimizer_dec1.step()

            loss_d2.backward()
            optimizer_dec2.step()


            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:

                # only pull stats when needed
                s_loss_d1 = loss_d1.detach()
                s_loss_d2 = loss_d2.detach()
                s_loss_sq = torch.mean(abs(loss_d1t - loss_d2t)).detach()


                temp1 = abs(dec1_out -dec2_out).detach()
                s_probDiff_mean = torch.mean(temp1)
                s_probDiff_std = torch.std(temp1)

                # just for reporting
                num0 = (y_fake<0.1).sum()
                num1 = (y_fake>0.9).sum()

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of samples
                z = Variable(Tensor(np.random.normal(0, 1, (len(x[0]), opt.latent_dim))))
                gen_sample = generator(z)
                # Adversarial loss

                fake_sample = gen_sample.clone()
                fake_wo_sen = gen_sample.clone()

                fake_wo_sen[:, 9:10] = 0
                y_fake = fake_sample[:, -1]

                loss_p1 = -torch.mean(discriminator(gen_sample))

                loss_d1 = loss_dec(dec1(fake_sample[:, :-1]).reshape(-1).float(), y_fake.float())
                loss_d2 = loss_dec(dec2(fake_wo_sen[:, :-1]).reshape(-1).float(), y_fake.float())
                loss_p2 = torch.mean(abs(loss_d1-loss_d2))

                loss_G = loss_p1 + opt.lam * loss_p2

                loss_G.backward()


                q1 = generator.parameters()
                s1 = list(q1)
                if torch.isnan(s1[0].grad).any():
                    import pdb
                    pdb.set_trace()

                optimizer_G.step()

                q1 = generator.parameters()
                s1 = list(q1)
                if torch.isnan(s1[0]).any():
                    import pdb
                    pdb.set_trace()


                if batches_done %200 ==0:
                    print(
                        "Lambda: %f [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [d1 loss: %f] [d2 loss: %f]  [sq d1 d2: %f]  [num0: %f] [num1 %f]"
                        % (
                            opt.lam, epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(),
                            loss_G.item(), s_loss_d1, s_loss_d2, s_loss_sq, num0,num1)
                    )
                if storeStats:
                    temp1 = {}
                    temp1["d1d2_form"] = "cross"
                    temp1["lam"] = float(opt.lam)
                    temp1["Epoch"] = float(epoch)
                    temp1["Batch"] = float(batches_done + epoch*len(dataloader))
                    temp1["D_loss"] = float(loss_D.item())
                    temp1["G_loss"] = float(loss_G.item())
                    temp1["d1_loss"] = float(s_loss_d1.numpy())
                    temp1["d2_loss"] = float(s_loss_d2.numpy())
                    temp1["d1d2_loss"] = float(s_loss_sq.numpy())
                    temp1["cur_d1d2"] = float(loss_p2.detach().numpy())

                    temp1["num0"] = float(num0.numpy())
                    temp1["num1"] = float(num1.numpy())

                    temp1["probDiff_mean"] = float(s_probDiff_mean.numpy())
                    temp1["probDiff_std"]  = float(s_probDiff_std.numpy())

                    stats_store.append(temp1)

            # if batches_done % opt.sample_interval == 0:
            #    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1

    z = Tensor(np.random.normal(0, 1, (X_in.shape[0], opt.latent_dim)))
    X_gen = generator(z)

    X_gen = X_gen.detach().numpy()

    if storeStats:
        return X_gen,pd.DataFrame(stats_store)
    return X_gen
