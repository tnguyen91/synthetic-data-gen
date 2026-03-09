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
            *block(opt.latent_dim, 20, normalize=False),
            *block(20, 10),
            nn.Linear(10, 8),
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
            nn.Linear(8, 20),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        # img_flat = img.view(img.shape[0], -1)
        validity = self.model(x)
        return validity


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(7, 10),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 20),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x)
        return y


def train_ifgan(X_in, opt):
    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator()
    dec1 = Decoder()
    dec2 = Decoder()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader

    X = torch.from_numpy(X_in.astype(np.float32))

    D = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(D, batch_size=opt.batch_size, shuffle=True, )
    # loss_dec = nn.CrossEntropyLoss()
    loss_dec = torch.nn.BCELoss()
    #loss_dec = nn.NLLLoss()

    # Optimizers
    # optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    # optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_dec1 = torch.optim.Adam(dec1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_dec2 = torch.optim.Adam(dec2.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # # Adding noise for differential privacy (note that, clipping also needed)
    # if sigma is not None:
    #     for parameter in discriminator.parameters():
    #         parameter.register_hook(
    #             lambda grad: grad + (1 / opt.batch_size) * sigma * torch.randn(parameter.shape)
    #         )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):

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
            fake_wo_sen = copy.deepcopy(fake_sample)
            fake_wo_sen[:, 5:7] = 0
            y_fake = fake_sample[:, -1]

    #        y_fake = y_fake >0.5
#            import pdb
#            pdb.set_trace()
#            y_fake = y_fake.type(torch.LongTensor)
            #y_fake = fake_sample[:, -1:]
            # Adversarial loss D tries to map fake to zero and real to one
            loss_D = -torch.mean(discriminator(real_sample)) + torch.mean(discriminator(fake_sample))

            loss_d1 = loss_dec(dec1(fake_sample[:, :-1]).float(), y_fake.reshape(-1,1).float())

            loss_d2 = loss_dec(dec2(fake_wo_sen[:, :-1]).float(), y_fake.reshape(-1,1).float())



            num0 = y_fake.tolist().count(0)
            num1 = y_fake.tolist().count(1)

            loss_D.backward()
            optimizer_D.step()

            loss_d1.backward()
            optimizer_dec1.step()
            s_loss_d1 = loss_d1.detach()

            loss_d2.backward()
            optimizer_dec2.step()
            s_loss_d2 = loss_d2.detach()

            s_loss_sq = (s_loss_d1-s_loss_d2)**2

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

                """
                fake_sample = gen_sample.detach()
                fake_wo_sen = copy.deepcopy(fake_sample)
                fake_wo_sen[:, 60:62] = 0
                y_fake = fake_sample[:, -1]
                y_fake = y_fake.type(torch.LongTensor)
                loss_d1 = loss_dec(dec1(fake_sample[:, :-1]), y_fake)
                loss_d2 = loss_dec(dec2(fake_wo_sen[:, :-1]), y_fake)
                """

                fake_sample = gen_sample.clone()
                fake_wo_sen = gen_sample.clone()
                fake_wo_sen[:, 5:7] = 0
                y_fake = fake_sample[:, -1]

#                y_fake = y_fake > 0.5

                #y_fake = y_fake.type(torch.LongTensor)

                loss_d1 = loss_dec(dec1(fake_sample[:, :-1]).reshape(-1).float(), y_fake.float())
                loss_d2 = loss_dec(dec2(fake_wo_sen[:, :-1]).reshape(-1).float(), y_fake.float())

                # loss_dec2 = torch.nn.BCELoss(reduction="none")
                # loss_d1 = loss_dec2(dec1(fake_sample[:, :-1]), y_fake)
                # loss_d2 = loss_dec2(dec2(fake_wo_sen[:, :-1]), y_fake)

                loss_G = -torch.mean(discriminator(gen_sample)) + opt.lam * torch.mean((loss_d1 - loss_d2)**2)

                loss_G.backward()


                optimizer_G.step()
                q1 = generator.parameters()
                s1 = list(q1)
                if torch.isnan(s1[0]).any():
                    import pdb
                    pdb.set_trace()


                print(
                    # "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [d1 loss: %f] [d2 loss: %f]  [sq d1 d2: %f]  [num0: %f] [num1 %f]"
                    % (
                        epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(),
                        loss_G.item(), s_loss_d1, s_loss_d2, s_loss_sq, num0,num1)
                )

            # if batches_done % opt.sample_interval == 0:
            #    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1

    z = Tensor(np.random.normal(0, 1, (X_in.shape[0], opt.latent_dim)))
    X_gen = generator(z)

    X_gen = X_gen.detach().numpy()
    return X_gen

# class Decoder1(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(104, 64),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.Linear(512, 256),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # nn.Linear(64, 20),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 1),
#         )
#
#     def forward(self, x, a):
#         # Concatenate label embedding and image to produce input
#         dis_in = torch.cat((a, x), -1)
#         y = self.model(dis_in)
#         return y
