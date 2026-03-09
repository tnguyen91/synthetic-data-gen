import argparse
import os
import numpy as np
import copy
import math
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

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
            nn.Linear(256, 105),
            # nn.Tanh()
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
            nn.Linear(105, 256),
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
            nn.Linear(2, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 20),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x)
        return y


def train_spgan(X_in, opt):
    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator()
    dec1 = Decoder()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader

    X = torch.Tensor(X_in)

    D = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(D, batch_size=opt.batch_size, shuffle=True, )

    loss_dec = torch.nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_dec1 = torch.optim.Adam(dec1.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):

        for i, x in enumerate(dataloader):

            real_sample = x[0]

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            optimizer_dec1.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (len(x[0]), opt.latent_dim))))

            # Generate a batch of images

            fake_sample = generator(z).detach()
            sen_att = fake_sample[:, 60:62]
            y_fake = fake_sample[:, -1]

            # Adversarial loss

            # train the discriminator
            loss_D = -torch.mean(discriminator(real_sample)) + torch.mean(discriminator(fake_sample))
            loss_D.backward()
            optimizer_D.step()

            # train the decoder
            loss_d1 = loss_dec(dec1(sen_att).float(), y_fake.reshape(-1, 1).float())
            loss_d1.backward()
            optimizer_dec1.step()


            # just for reporting
            num0 = (y_fake<0.1).sum()
            num1 = (y_fake>0.9).sum()

            s_loss_d1 = loss_d1.detach()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of samples
                z = Variable(Tensor(np.random.normal(0, 1, (len(x[0]), opt.latent_dim))))
                gen_sample = generator(z)
                # Adversarial loss


                fake_sample = gen_sample.clone()
                sen_att = fake_sample[:, 60:62]
                y_fake = fake_sample[:, -1]


                loss_d1 = loss_dec(dec1(sen_att).reshape(-1).float(), y_fake.float())
                loss_G = -torch.mean(discriminator(gen_sample)) - opt.lam * torch.mean(loss_d1)

                loss_G.backward()
                optimizer_G.step()

                print(
                    # "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [d1 loss: %f] [num0: %f] [num1 %f]"
                    % (
                        epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(),
                        loss_G.item(), s_loss_d1, num0, num1)
                )

            # if batches_done % opt.sample_interval == 0:
            #    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1

    z = Tensor(np.random.normal(0, 1, (X_in.shape[0], opt.latent_dim)))
    X_gen = generator(z)

    X_gen = X_gen.detach().numpy()

    return X_gen
