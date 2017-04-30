from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

import models.dcgan as dcgan
import models.mlp as mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nconv', type=int, default=64)
parser.add_argument('--nef', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--nstart', type=int, default=25, help='number of startup generator runs')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--Diter_plus', type=int, default=200, help='number of D iters per each G iter for "large" phases')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nconv = int(opt.nconv)
nef = int(opt.nef)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, n_extra_layers)
netG.apply(weights_init)
#if opt.netG != '': # load checkpoint if needed
#    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netE = dcgan.DCGAN_E(opt.imageSize, nz, nc, nef, n_extra_layers)
netE.apply(weights_init)
#if opt.netE != '':
#    netE.load_state_dict(torch.load(opt.netE))
print(netE)

nx = opt.imageSize**2 * nc
if opt.mlp_D:
    #print('using MLP discriminator')
    netD = mlp.MLP_D(nx + netE.nz, ndf)
else:
    # Use a DCGAN to turn the image to a vector, then compare against z.
    print('using DCNN->MLP discriminator')
    netD = mlp.MLP_ED(opt.imageSize, nz, nc, ndf, nconv=nconv)
    netD.netE.apply(weights_init)
    netD.netG.apply(weights_init)
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
# Fixed z-vector that we can compare as training progresses.
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    print('Moving models to GPU')
    netD.cuda()
    netG.cuda()
    netE.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(list(netG.parameters()) + list(netE.parameters()),
            lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(list(netG.parameters()) + list(netE.parameters()),
            lr = opt.lrG)

def rcpu(dataloader):
    for data in dataloader:
        real_cpu, _ = data
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        yield real_cpu

gen_iterations = 0
print('Starting training')
for epoch in range(opt.niter):
    print('Starting epoch:', epoch)
    data_iter = rcpu(dataloader)
    i = 0
    redo = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations == 0:
            Diters = 1 # Self-test
        elif redo > 0 or gen_iterations < opt.nstart or gen_iterations % 500 == 0:
            Diters = opt.Diter_plus
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            real = data_iter.next()
            i += 1

            # train with real
            netD.zero_grad()
            batch_size = real.size(0)
            input.resize_as_(real).copy_(real)
            inputz = Variable(netE(Variable(input, volatile=True)).data)
            inputv = Variable(input)

            # Encoded z, Real x
            errD_real = netD(inputz, inputv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev).data)
            inputz = Variable(noise).squeeze()
            inputv = fake
            # Real z, Decoded x
            errD_fake = netD(inputz, inputv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        # Hack to keep training critic if distance too low.
        if redo < 10 and errD.abs().data[0] < 0.000001:
            if redo == 0:
                print(errD.abs().data[0])
            redo += 1
            continue
        if redo > 0:
            redo = -3

        if i == len(dataloader):
            break

        ############################
        # (2) Update G, E network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        netE.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        # Real z, Decoded x
        errG = netD(noisev.squeeze(), fake)
        errG.backward(one)

        real = data_iter.next()
        i += 1
        # train with real
        batch_size = real.size(0)
        input.resize_as_(real).copy_(real)
        inputv = Variable(input)
        fakez = netE(inputv)
        # Encoded z, Real x
        errE = netD(fakez, inputv)
        errE.backward(mone)

        errB = errG - errE
        optimizerG.step()
        gen_iterations += 1

        if gen_iterations < opt.nstart or gen_iterations % 50 == 0:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_B: %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                errD.data[0], errB.data[0]))
        if gen_iterations == 1 or gen_iterations % 500 == 0:
            # Generate reconstructions of real samples as well as
            # Images from a fixed noise vector.
            real_z = netE(Variable(real, volatile=True))
            real = real.mul(0.5).add(0.5)
            vutils.save_image(real, '{0}/real_samples_{1}.png'.format(
                opt.experiment, gen_iterations))
            fake = netG(Variable(fixed_noise, volatile=True))
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(
                opt.experiment, gen_iterations))
            reconst = netG(real_z.unsqueeze(2).unsqueeze(3))
            reconst.data = reconst.data.mul(0.5).add(0.5)
            vutils.save_image(reconst.data, '{0}/reconstruction_samples_{1}.png'.format(
                opt.experiment, gen_iterations))

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netE.state_dict(), '{0}/netE_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
