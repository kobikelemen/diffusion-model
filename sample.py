import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from unet import UNet

def denorm(x, channels=None, w=None ,h=None, resize = False):
    # mean = torch.Tensor([0.4914, 0.4822, 0.4465])
    # std = torch.Tensor([0.247, 0.243, 0.261])
    mean = torch.Tensor([0.4914])
    std = torch.Tensor([0.247])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    x = unnormalize(x)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


def sample_simple_diffusion_model(model):
    data_path = './data'
    batch_size = 100
    transform = transforms.Compose([transforms.ToTensor(),])
    test_dat = datasets.MNIST(data_path, train=False, transform=transform)
    loader_test = DataLoader(test_dat, batch_size, shuffle=False)
    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[0:32, :, :, :]
    img = make_grid(denorm(fixed_input), nrow=8, padding=2, normalize=False,
                    value_range=None, scale_each=False, pad_value=0)
    plt.figure()
    # show(img)
    plt.savefig('/home/kk2720/dl/diffusion-model/plots/reference_mnist.jpeg')


if __name__ == "__main__":
    model_path = '/home/kk2720/dl/diffusion-model/model/mnist_simple_diffusion.pt'
    model = UNet(1000,100)
    model.load_state_dict(torch.load(model_path))
    sample_simple_diffusion_model(model)

    