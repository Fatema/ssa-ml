# main imports
import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import re

# local version imports
import visdom
vis = visdom.Visdom(server='ncc1.clients.dur.ac.uk',port=12345)
vis.line(X=np.array([0]), Y=np.array([[np.nan,np.nan,np.nan]]), win='loss')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ENCODER_MODEL_PATH = 'encoder_model.pkl'
DECODER_MODEL_PATH = 'decoder_model.pkl'
DISCRIMINATOR_MODEL_PATH = 'discriminator_model.pkl'

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#transforms 
input_shape = 100
rotation_degrees = 30
scale = 32
mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)

training_transforms = transforms.Compose([
    # transforms.RandomRotation(rotation_degrees),
    # transforms.RandomResizedCrop(input_shape),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(scale),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

testing_transforms = transforms.Compose([
    transforms.Resize(scale),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True, transform=training_transforms),
    shuffle=True, batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, download=True, transform=testing_transforms),
    shuffle=False, batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# define variational autoencoder components along with a discriminator models
class Encoder(nn.Module):
    def __init__(self, intermediate_size=128, hidden_size=20):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)

        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

    def encode(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1  = F.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, intermediate_size=128, hidden_size=20):
        super(Decoder, self).__init__()

        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def decode(self, z):
        h3  = F.relu(self.fc3(z))
        out = F.relu(self.fc4(h3))
        out = out.view(out.size(0), 32, 16, 16)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = torch.sigmoid(self.conv5(out))
        return out

    def forward(self, z):
        return self.decode(z)

class ResidualBlock(nn.Module):
   def __init__(self, in_features):
       super(ResidualBlock, self).__init__()

       conv_block = [ nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(in_features) ]

       self.conv_block = nn.Sequential(*conv_block)

   def forward(self, x):
       return x + self.conv_block(x)

class Discriminator(nn.Module):
    def __init__(self, f=64):
        super(Discriminator, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(3, f, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(f, f*2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(f*2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(f*2, f*4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(f*4))
        layers.append(ResidualBlock(f*4))
        layers.append(ResidualBlock(f*4))
        layers.append(ResidualBlock(f*4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(f*4, f*8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(f*8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(f*8, 1, 4, 2, 1, bias=False))
        layers.append(nn.Sigmoid())
        self.layers = layers

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

N_Encoder = Encoder().to(device)
N_Decoder = Decoder().to(device)
N_Discriminator = Discriminator().to(device)

try:
    N_Encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH))
    N_Encoder.eval()
    N_Decoder.load_state_dict(torch.load(DECODER_MODEL_PATH))
    N_Decoder.eval()
    N_Discriminator.load_state_dict(torch.load(DISCRIMINATOR_MODEL_PATH))
    N_Discriminator.eval()
except:
    print('no model found')

optimiser_encoder = torch.optim.Adam(N_Encoder.parameters(), lr=0.001)
optimiser_decoder = torch.optim.Adam(N_Decoder.parameters(), lr=0.001)
optimiser_discriminator = torch.optim.Adam(N_Discriminator.parameters(), lr=0.001)

bce_loss = nn.BCELoss()

epoch = 0

# VAE loss has a reconstruction term and a KL divergence term summed over all elements and the batch
def vae_loss(p, x, mu, logvar):
    BCE = F.binary_cross_entropy(p.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD

lambda_bce = 1e-6

# main training loop
# training loop
while (epoch < 100):
    
    # arrays for metrics
    logs = {}
    train_loss_encoder_arr = np.zeros(0)
    train_loss_decoder_arr = np.zeros(0)
    train_loss_discriminator_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(1000):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        optimiser_encoder.zero_grad()
        optimiser_decoder.zero_grad()
        optimiser_discriminator.zero_grad()
        
        z, mu, logvar = N_Encoder(x)
        p = N_Decoder(z)

        loss_bce, loss_kl = vae_loss(p, x, mu, logvar)

        loss_encoder = torch.sum(loss_kl) + torch.sum(loss_bce)
        l_r = bce_loss(N_Discriminator.discriminate(x).mean(), torch.ones(1)[0].to(device)) # real -> 1
        l_f = bce_loss(N_Discriminator.discriminate(p).mean(), torch.zeros(1)[0].to(device)) #  fake -> 0
        loss_discriminator = (l_r + l_f)/2.0

        loss_decoder = torch.sum(lambda_bce * loss_bce) - (1 - loss_bce) * loss_discriminator

        loss_encoder.backward(retain_graph=True)
        loss_decoder.backward(retain_graph=True)
        loss_discriminator.backward(retain_graph=True)

        optimiser_encoder.step()
        optimiser_decoder.step()
        optimiser_discriminator.step()

        train_loss_encoder_arr = np.append(train_loss_encoder_arr, loss_encoder.cpu().data)
        train_loss_decoder_arr = np.append(train_loss_decoder_arr, loss_decoder.cpu().data)
        train_loss_discriminator_arr = np.append(train_loss_discriminator_arr, loss_discriminator.cpu().data)

    example_1 = (test_loader.dataset[13][0]).to(device)  # horse
    example_2 = (test_loader.dataset[160][0]).to(device) # bird

    ex1_z, ex1_mu, ex1_logvar = N_Encoder(example_1.unsqueeze(0))
    ex2_z, ex2_mu, ex2_logvar = N_Encoder(example_2.unsqueeze(0))

    # this is some sad blurry excuse of a Pegasus, hopefully you can make a better one
    bad_pegasus = N_Decoder(0.9*ex1_z + 0.1*ex2_z).squeeze(0)

    pegasus = bad_pegasus.cpu().data.permute(0,2,1).contiguous().permute(2,1,0)

    vis.image(
        pegasus.numpy().T
    )

    torch.save(copy.deepcopy(N_Encoder.state_dict()), 'encoder_model.pkl')
    torch.save(copy.deepcopy(N_Decoder.state_dict()), 'decoder_model.pkl')
    torch.save(copy.deepcopy(N_Discriminator.state_dict()), 'discriminator_model.pkl')

    # plot metrics
    vis.line(X=np.array([epoch]), Y=np.array([[
        train_loss_encoder_arr.mean(),
        train_loss_decoder_arr.mean(),
        train_loss_discriminator_arr.mean(),
    ]]), win='loss', opts=dict(title='loss', xlabel='epoch', ylabel='loss', ytype='log', legend=[
        'encoder loss',
        'decoder loss',
        'discriminator loss'
    ]), update='append')

    epoch = epoch + 1

example_1 = (test_loader.dataset[13][0]).to(device)  # horse
example_2 = (test_loader.dataset[160][0]).to(device) # bird

ex1_z, ex1_mu, ex1_logvar = N_Encoder(example_1.unsqueeze(0))
ex2_z, ex2_mu, ex2_logvar = N_Encoder(example_2.unsqueeze(0))

# this is some sad blurry excuse of a Pegasus, hopefully you can make a better one
bad_pegasus = N_Decoder(0.9*ex1_z + 0.1*ex2_z).squeeze(0)

pegasus = bad_pegasus.cpu().data.permute(0,2,1).contiguous().permute(2,1,0)

vis.image(
    pegasus.numpy().T
)