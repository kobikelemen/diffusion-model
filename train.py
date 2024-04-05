import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from math import sqrt
from unet.unet import OneResUNet
from diffusion_model import Trainer

NUM_GPUS = 4



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def calc_beta(min_b, max_b, t, max_t):
    beta = min_b + (t/max_t) * (max_b - min_b) 
    return beta


def calc_lists(min_b, max_b, num_steps, rank):
    at_hat_ = 1
    beta_list_, at_list_, at_hat_list_ = [], [], []
    for t in range(num_steps):
        b = calc_beta(min_b, max_b, t, num_steps)
        beta_list_.append(b)
        at_list_.append(1-b)
        at_hat_ *= (1-b)
        at_hat_list_.append(at_hat_)
    return torch.tensor(beta_list_).to(f'cuda:{rank}'), torch.tensor(at_list_).to(f'cuda:{rank}'), torch.tensor(at_hat_list_).to(f'cuda:{rank}')


def prepare_train(rank, world_size, batch_size, data_path, transform, pin_memory=False, num_workers=0):
    train_dat = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_dat, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dat, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, generator=torch.Generator(device=f'cuda:{rank}'), sampler=sampler)
    return train_dataloader


def prepare_test(rank, world_size, batch_size, data_path, transform, pin_memory=False, num_workers=0):
    test_dat = datasets.CIFAR10(data_path, train=False, transform=transform)
    sampler = DistributedSampler(test_dat, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dat, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, generator=torch.Generator(device=f'cuda:{rank}'), sampler=sampler)
    return test_dataloader


def train(rank, world_size):
    setup(rank, world_size)
    torch.set_default_device(f'cuda:{rank}')

    learning_rate = 2e-4
    num_epochs = 2000
    batch_size = 100
    min_beta = 10**-4
    max_beta = 0.02
    timesteps = 1000
    img_w = 32
    img_h = 32
    img_c = 3
    data_path = './data'
    beta_list, at_list, at_hat_list = calc_lists(min_beta, max_beta, timesteps, rank)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x - 0.5) * 2)])
    loader_train = prepare_train(rank, world_size, batch_size, data_path, transform)
    loader_test = prepare_test(rank, world_size, batch_size, data_path, transform)
    model = OneResUNet(32, channels=img_c).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    trainer = Trainer(batch_size, timesteps, num_epochs, model, optimizer, loader_test, loader_train, img_c, img_w, img_h, rank, at_hat_list)
    train_loss, test_loss = trainer.train()


    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.ylabel('Epochs')
    plt.savefig('/home/kk2720/dl/diffusion-model/plots/cifar_diffusion_loss1.jpeg')
    cleanup()

if __name__ == '__main__':
    mp.spawn(train, args=(NUM_GPUS,), nprocs=NUM_GPUS)