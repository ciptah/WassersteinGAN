import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_E(nn.Module):
    def __init__(self, isize, nz, nc, nef, n_extra_layers=0):
        super(DCGAN_E, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()

        cnef = nc
        main.add_module('preproc.conv',
                        nn.Conv2d(nc, nef, 3, 1, 1, bias=False))
        main.add_module('preproc.{0}.relu'.format(cnef),
                        nn.LeakyReLU(0.2, inplace=True))
        cnef = nef

        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, nef),
                        nn.Conv2d(nef, nef, 4, 2, 1, bias=False))
        main.add_module('preproc.batchnorm',
                        nn.BatchNorm2d(cnef))
        main.add_module('initial.relu.{0}'.format(nef),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cnef = isize / 2, nef

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cnef),
                            nn.Conv2d(cnef, cnef, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cnef),
                            nn.BatchNorm2d(cnef))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cnef),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cnef
            out_feat = cnef * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cnef = cnef * 2
            csize = csize / 2

        # state size. K x 4 x 4
        assert csize == 4
        main.add_module('final.{0}-{1}.conv'.format(cnef, 1),
                        nn.Conv2d(cnef, cnef, 4, 1, 0, bias=False))
        self.final_linear = nn.Linear(cnef, nz)
        self.main = main
        self.nz = nz

    def forward(self, x, predict=False):
        output = self.main(x).view(x.size(0), -1) # cnef
        return self.final_linear(output) # compress to Z

class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, cngf),
                        nn.ConvTranspose2d(cngf, cngf, 4, 2, 1, bias=False))
        main.add_module('final-{0}.relu'.format(cngf),
                        nn.ReLU(True))
        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 3, 1, 1, bias=False))

        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, z):
        output = self.main(z)
        return output 

