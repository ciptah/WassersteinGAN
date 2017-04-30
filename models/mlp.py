from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import models.dcgan as dg


class MLP_D(nn.Module):
    def __init__(self, n, ndf):
        super(MLP_D, self).__init__()

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(n, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main

    def forward(self, input):
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)

class MLP_ED(nn.Module):
    '''Combines DCGAN and MLP.'''
    def __init__(self, isize, nz, nc, ndf, nconv=0):
        super(MLP_ED, self).__init__()

        if nconv == 0:
            nconv = ndf

        self.netG = dg.DCGAN_G(isize, nz, nc, nconv)
        self.netE = dg.DCGAN_E(isize, nz, nc * 2, nconv, linear_growth=True)
        self.pool = nn.AvgPool2d(2)

        all_dim = 2 * nz
        self.mlp = MLP_D(all_dim, ndf)
        self.nz = nz

    def forward(self, z, x):
        xt = self.netG(z.unsqueeze(2).unsqueeze(3))
        zt = self.netE(torch.cat([x, xt], 1))
        all_ins = torch.cat([z, zt], 1)
        return self.mlp(all_ins)
