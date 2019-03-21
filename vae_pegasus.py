# main imports
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# local version imports
import visdom
vis = visdom.Visdom(server='ncc1.clients.dur.ac.uk',port=12345)
vis.line(X=np.array([0]), Y=np.array([[np.nan, np.nan]]), win='loss')

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

# define variational autoencoder model
class VAE(nn.Module):
    def __init__(self, intermediate_size=128, hidden_size=20):
        super(VAE, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)

        # latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # decoder
        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

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

    def decode(self, z):
        h3  = F.relu(self.fc3(z))
        out = F.relu(self.fc4(h3))
        out = out.view(out.size(0), 32, 16, 16)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = torch.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


N = VAE().to(device)

optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
epoch = 0

# VAE loss has a reconstruction term and a KL divergence term summed over all elements and the batch
def vae_loss(p, x, mu, logvar):
    BCE = F.binary_cross_entropy(p.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

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
        
        p, mu, logvar = N(x)
        loss = vae_loss(p, x, mu, logvar)
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    # plot metrics
    vis.line(X=np.array([epoch]), Y=np.array([[
        train_loss_arr.mean()
    ]]), win='loss', opts=dict(title='loss',xlabel='epoch', ylabel='loss', ytype='log', legend=[
        'train loss',
    ]), update='append')

    epoch = epoch + 1


example_1 = transforms.ToTensor()(test_loader.dataset.test_data[13]).to(device)  # horse
example_2 = transforms.ToTensor()(test_loader.dataset.test_data[160]).to(device) # bird

example_1_code = N.encode(example_1.unsqueeze(0))
example_2_code = N.encode(example_2.unsqueeze(0))

# this is some sad blurry excuse of a Pegasus, hopefully you can make a better one
bad_pegasus = N.decode(0.9*example_1_code + 0.1*example_2_code).squeeze(0)

plt.grid(False)
viz.matplot(plt.imshow(bad_pegasus.cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary))