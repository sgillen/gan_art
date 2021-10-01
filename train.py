import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np 

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
from torchvision.datasets import DatasetFolder
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
from torchvision.utils import save_image

image_size = 128
batch_size = 16
latent_size = 256

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def tuple_load(file_name):
    tup = torch.load(file_name)
    return tup[0]

def save_samples(index, latent_tensors, show=True):
    fake_images = gen(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = dis(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = gen(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = dis(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = gen(latent)
    
    # Try to fool the discriminator
    preds = dis(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()



if __name__ == "__main__":
    import torch.nn.functional as F
    import time
    
    device = 'cuda:0'
    #device='cpu'
    train_ds = DatasetFolder('./wikiart_preprocessed_128/', loader=tuple_load, extensions='.pkl') 
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    train_dl_gpu = DeviceDataLoader(train_dl, device)

    dis = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(1024, 1, kernel_size=4, stride=4, padding=1, bias=False),
        nn.Flatten(),
        nn.Sigmoid()
    )
    
    gen = nn.Sequential(
        nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(128,   3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        
    )
    
        
    epochs = 100
    lr = 0.0002
    start_idx= 0 

    # Uncomment this to refine a policy
    #gen = torch.load("./model_128_2/gen_v12_145")
    #dis = torch.load("./model_128_2/dis_v12_145")    
    # epochs = 300
    # lr = 0.00005
    # start_idx= 101 

    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))


    for epoch in range(start_idx, epochs):
        start = time.time()
        for real_images, _ in train_dl_gpu:
            #print(i, train_dl.dl.dataset.samples[i])
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator#         
            loss_g = train_generator(opt_g)
            #print(f"during epoch {epoch}: {torch.cuda.max_memory_allocated()}")
        print(f"epoch: {epoch} time (minutes): {(time.time() - start)/60}")


        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

        if epoch % 5 == 0:
            torch.save(gen, f"./model_128_2/gen_v12_{epoch}")
            torch.save(dis, f"./model_128_2/dis_v12_{epoch}")
            

