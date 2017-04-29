from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn


class MLP_D(nn.Module):
    def __init__(self, isize, nc, nz, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize + nz, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize

    def forward(self, inputz, inputv):
        inputv = inputv.view(inputv.size(0), -1)
        input = torch.cat([inputz, inputv], 0)

        output = self.main(input)
        output = output.mean(0)
        return output.view(1)
