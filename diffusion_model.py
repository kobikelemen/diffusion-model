import random
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

from unet import UNet

learning_rate = 1e-3
num_epochs = 10
batch_size = 20
# beta = 0.01
# min_beta = 10**-4
# max_beta = 0.02
min_beta = 0.01
max_beta = 0.01
timesteps = 1000
img_w = 28
img_h = 28
data_path = './data'
device = 'cuda:0'

def calc_beta(min_b, max_b, t, max_t):
    beta = min_b + (t/max_t) * (max_b - min_b) 
    return beta

def calc_lists(min_b, max_b, num_steps):
    at_hat_ = 1
    beta_list_, at_list_, at_hat_list_ = [], [], []
    for t in range(timesteps):
        b = calc_beta(min_b, max_b, t, num_steps)
        beta_list_.append(b)
        at_list_.append(1-b)
        at_hat_ *= (1-b)
        at_hat_list_.append(at_hat_)
    return torch.tensor(beta_list_).to(device), torch.tensor(at_list_).to(device), torch.tensor(at_hat_list_).to(device)


beta_list, at_list, at_hat_list = calc_lists(min_beta, max_beta, timesteps)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - 0.5) * 2)
])

train_dat = datasets.MNIST(data_path, train=True, download=True, transform=transform)
test_dat = datasets.MNIST(data_path, train=False, transform=transform)

loader_train = DataLoader(train_dat, batch_size, shuffle=True, generator=torch.Generator(device=device))
loader_test = DataLoader(test_dat, batch_size, shuffle=False, generator=torch.Generator(device=device))





def calc_xt(epsilon, t, x0):
    """ Note: this assumes beta doesn't vary with time step """
    # at_hat = 1
    # for s in range(1,t[0]+1):
    #     at_hat *= (1 - calc_beta(min_beta, max_beta, s, timesteps))
    at_hat = at_hat_list[t]
    # return torch.sqrt(at_hat) * x0 + torch.sqrt(1 - at_hat) * epsilon
    return torch.sqrt(at_hat.view(batch_size,1,1,1)) * x0 + torch.sqrt(1 - at_hat.view(batch_size,1,1,1)) * epsilon

def sample_t(batch_size, timesteps):
    # t = random.randint(1,timesteps)
    # return torch.full((batch_size,), t)
    return torch.randint(1, timesteps, (batch_size,))

def sample_epsilon(batch_size, img_h, img_w):
    return torch.normal(torch.zeros((batch_size, 1, img_h, img_w)), torch.ones((batch_size, 1, img_h, img_w)))

def calc_loss(epsilon, epsilon_pred):
    return torch.linalg.vector_norm(epsilon - epsilon_pred)


torch.set_default_device(device)
model = UNet(num_steps=timesteps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))



train_loss, test_loss = [], []
min_test_loss = 10e10

for epoch in range(num_epochs):
    epoch_train_loss = 0
    epoch_test_loss = 0
    mse = nn.MSELoss()
    with tqdm.tqdm(loader_train, unit="batch") as tepoch:
        for batch_idx, (data, _) in enumerate(tepoch):
            data = data.to(device)
            epsilon = sample_epsilon(batch_size, img_h, img_w)
            t = sample_t(batch_size, timesteps)
            xt = calc_xt(epsilon, t, data)
            epsilon_pred = model(xt, t)
            loss = mse(epsilon, epsilon_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item() * len(data) / len(loader_train.dataset)
            if batch_idx % 20 == 0:
                tepoch.set_description(f"Train Epoch {epoch}")
                tepoch.set_postfix(loss=epoch_train_loss)
        print(f'Train Loss: {epoch_train_loss}')
        train_loss.append(epoch_train_loss)

        with tqdm.tqdm(loader_test, unit="batch") as tepoch:
            for batch_idx, (data, _) in enumerate(tepoch):
                data = data.to(device)
                with torch.no_grad():
                    epsilon = sample_epsilon(batch_size, img_h, img_w)
                    t = sample_t(batch_size, timesteps)
                    xt = calc_xt(epsilon, t, data)
                    epsilon_pred = model(xt, t)
                    loss = mse(epsilon, epsilon_pred)
                    epoch_test_loss += loss.item() * len(data) / len(loader_test.dataset)
                    if batch_idx % 20 == 0:
                        tepoch.set_description(f"Train Epoch {epoch}")
                        tepoch.set_postfix(loss=epoch_test_loss)
            print(f'Test Loss: {epoch_test_loss}')
            test_loss.append(epoch_test_loss)
        

        if epoch_test_loss < min_test_loss:
            torch.save(model.state_dict(), '/home/kk2720/dl/diffusion-model/model/mnist_simple_diffusion12.pt')
            min_test_loss = epoch_test_loss


epochs = [i for i in range(num_epochs)]
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, test_loss, label='Test Loss')
plt.title('Loss')
plt.ylabel('Epochs')
plt.savefig('/home/kk2720/dl/diffusion-model/plots/simple_diffusion_loss12.jpeg')