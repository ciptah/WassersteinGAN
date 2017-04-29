from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn


class MLP_D(nn.Module):
    def __init__(self, nx, nz, ndf):
        super(MLP_D, self).__init__()

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nx + nz, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main

    def forward(self, inputz, inputv):
        inputv = inputv.view(inputv.size(0), -1).squeeze()
        input = torch.cat([inputz, inputv], 1)

        output = self.main(input)
        output = output.mean(0)
        return output.view(1)

class MLP_ED(nn.Module):
    '''Combines DCGAN and MLP.'''
    def __init__(self, netE, nz, ndf):
        super(MLP_ED, self).__init__()
        self.netE = netE
        self.mlp = MLP_D(self.netE.nz, netE.nz, ndf)
        self.nz = nz
    def forward(self, inputz, inputv):
        inputzz = self.netE(inputv)
        return self.mlp(inputzz, inputz)
