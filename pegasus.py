# main imports
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# local version imports
import visdom
vis = visdom.Visdom(server='ncc1.clients.dur.ac.uk',port=12345)
vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='loss')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    transforms.RandomRotation(rotation_degrees),
    transforms.RandomResizedCrop(input_shape),
    transforms.RandomHorizontalFlip(),
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

# define the model (a simple autoencoder)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features=3*32*32, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=32, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=3*32*32))
        layers.append(nn.Sigmoid())
        self.layers = layers

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    # encode (flatten as linear, then run first half of network)
    def encode(self, x):
        x = x.view(x.size(0), -1)
        for i in range(4):
            x = self.layers[i](x)
        return x

    # decode (run second half of network then unflatten)
    def decode(self, x):
        for i in range(4,8):
            x = self.layers[i](x)
        x = x.view(x.size(0), 3, 32, 32)
        return x

N = MyNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
epoch = 0

# main training loop
# training loop
while (epoch < 100):
    
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(1000):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        optimiser.zero_grad()
        p = N(x)
        loss = ((p-x)**2).mean() # simple l2 loss
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    # plot metrics
    vis.line(X=np.array([epoch]), Y=np.array([[
        train_loss_arr.mean()
    ]]), win='loss', opts=dict(title='loss',xlabel='epoch', ylabel='loss', ytype='log', legend=[
        'train loss'
    ]), update='append')

    epoch = epoch + 1


example_1 = (test_loader.dataset[13][0]).to(device)  # horse
example_2 = (test_loader.dataset[160][0]).to(device) # bird

example_1_code = N.encode(example_1.unsqueeze(0))
example_2_code = N.encode(example_2.unsqueeze(0))

# this is some sad blurry excuse of a Pegasus, hopefully you can make a better one
bad_pegasus = N.decode(0.9*example_1_code + 0.1*example_2_code).squeeze(0)

pegasus = bad_pegasus.cpu().data.permute(0,2,1).contiguous().permute(2,1,0)

vis.image(
    pegasus.numpy().T
)

