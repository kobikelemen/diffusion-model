import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
from math import sqrt

learning_rate = 1e-3
num_epochs = 20
batch_size = 100
beta = 0.001

data_path = './data'
device = 'cuda:0'

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dat = datasets.MNIST(
    data_path, train=True, download=True, transform=transform
)
test_dat = datasets.MNIST(data_path, train=False, transform=transform)

loader_train = DataLoader(train_dat, batch_size, shuffle=True, generator=torch.Generator(device=device))
loader_test = DataLoader(test_dat, batch_size, shuffle=False, generator=torch.Generator(device=device))


def make_time_embedding(num_steps, emb_dim):
    embedding_mat = torch.empty((num_steps, emb_dim))
    for step in range(num_steps):
        embedding = torch.tensor([1 / 10000 ** (2*i / emb_dim) for i in range(emb_dim)])
        if step % 2 == 0:
            embedding = (embedding * step).sin()
        else:
            embedding = (embedding * step).cos()
        embedding_mat[step] = embedding
    time_embed = nn.Embedding(num_steps, emb_dim)
    time_embed.weight.data = embedding_mat
    time_embed.requires_grad_(False)
    return time_embed



def make_time_embedding_mlp(in_d, out_d):
    return nn.Sequential(
        nn.Linear(in_d, out_d),
        nn.ReLU(),
        nn.Linear(out_d, out_d)
    )
        

class Block(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, normalize=True):
        super(Block, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU()
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    

class UNet(nn.Module):
    """ Using a UNet architecture as noise function approximator """
    def __init__(self, num_steps=1000, emb_dim=100):
        super(UNet, self).__init__()
        self.time_embed = make_time_embedding(num_steps, emb_dim)
        self.te1 = make_time_embedding_mlp(emb_dim, 1)
        self.b1 = nn.Sequential(
            Block((1, 28, 28), 1, 10),
            Block((10, 28, 28), 10, 10),
            Block((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = make_time_embedding_mlp(emb_dim, 10)
        self.b2 = nn.Sequential(
            Block((10, 14, 14), 10, 20),
            Block((20, 14, 14), 20, 20),
            Block((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = make_time_embedding_mlp(emb_dim, 20)
        self.b3 = nn.Sequential(
            Block((20, 7, 7), 20, 40),
            Block((40, 7, 7), 40, 40),
            Block((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = make_time_embedding_mlp(emb_dim, 40)
        self.b_mid = nn.Sequential(
            Block((40, 3, 3), 40, 20),
            Block((20, 3, 3), 20, 20),
            Block((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = make_time_embedding_mlp(emb_dim, 80)
        self.b4 = nn.Sequential(
            Block((80, 7, 7), 80, 40),
            Block((40, 7, 7), 40, 20),
            Block((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = make_time_embedding_mlp(emb_dim, 40)
        self.b5 = nn.Sequential(
            Block((40, 14, 14), 40, 20),
            Block((20, 14, 14), 20, 10),
            Block((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = make_time_embedding_mlp(emb_dim, 20)
        self.b_out = nn.Sequential(
            Block((20, 28, 28), 20, 10),
            Block((10, 28, 28), 10, 10),
            Block((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)
    
    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)
        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)
        out = self.conv_out(out)
        return out




def calc_xt(epsilon, beta, t, x0):
    """ Note: this assumes beta doesn't vary with time step """
    at = (1 - beta) ** t
    # print(f'at: {at.shape} x0: {x0.shape} epsilon: {epsilon.shape}')
    return torch.sqrt(at.view(batch_size,1,1,1)) * x0 + torch.sqrt(1 - at.view(batch_size,1,1,1)) * epsilon

def sample_t(batch_size, timesteps):
    return torch.randint(0, timesteps, (batch_size,))

def sample_epsilon(batch_size, img_h, img_w):
    return torch.normal(torch.zeros((batch_size, 1, img_h, img_w)), torch.ones((batch_size, 1, img_h, img_w)))

def calc_loss(epsilon, epsilon_pred):
    return torch.linalg.vector_norm(epsilon - epsilon_pred)

timesteps = 1000
img_w = 28
img_h = 28

torch.set_default_device(device)
model = UNet(num_steps=timesteps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))



train_loss, test_loss = [], []

for epoch in range(num_epochs):
    epoch_train_loss = 0
    epoch_test_loss = 0
    # if epoch > 1:
    #     break 

    with tqdm.tqdm(loader_train, unit="batch") as tepoch:
        for batch_idx, (data, _) in enumerate(tepoch):
            data = data.to(device)
            epsilon = sample_epsilon(batch_size, img_h, img_w)
            t = sample_t(batch_size, timesteps)
            xt = calc_xt(epsilon, beta, t, data)
            epsilon_pred = model(xt, t)
            loss = calc_loss(epsilon, epsilon_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()
            if batch_idx % 20 == 0:
                tepoch.set_description(f"Train Epoch {epoch}")
                tepoch.set_postfix(loss=epoch_train_loss/((batch_idx+1) * batch_size))
        print(f'Train Loss: {epoch_train_loss / (len(tepoch) * batch_size)}')
        train_loss.append(epoch_train_loss / (len(tepoch) * batch_size))

        with tqdm.tqdm(loader_test, unit="batch") as tepoch:
            for batch_idx, (data, _) in enumerate(tepoch):
                data = data.to(device)
                with torch.no_grad():
                    epsilon = sample_epsilon(batch_size, img_h, img_w)
                    t = sample_t(batch_size, timesteps)
                    xt = calc_xt(epsilon, beta, t, data)
                    epsilon_pred = model(xt, t)
                    loss = calc_loss(epsilon, epsilon_pred)
                    epoch_test_loss += loss.item()
            print(f'Test Loss: {epoch_test_loss / (len(tepoch) * batch_size)}')
            test_loss.append(epoch_test_loss / (len(tepoch) * batch_size))
        


torch.save(model.state_dict(), '/home/kk2720/dl/diffusion-model/model/mnist_simple_diffusion.pt')


epochs = [i for i in range(num_epochs)]
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, test_loss, label='Test Loss')
plt.title('Loss')
plt.ylabel('Epochs')
plt.savefig('/home/kk2720/dl/diffusion-model/plots/simple_diffusion_loss.jpeg')