import torch
import torch.nn as nn
import numpy as np 

import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from PIL import Image,ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True



import os

image_size = 128

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder('./wikiart/', 
                       transform=tt.Compose([
                           tt.Resize(image_size),
                           tt.CenterCrop(image_size),
                           tt.ToTensor(),
                           tt.Normalize(*stats)]))

for i in range(len(train_ds)):
    path = train_ds.samples[i][0]
    tt_tensor = train_ds[i]
    
    genre = path.split('/')[2]
    name = path.split('/')[-1].split('.')[0]
    save_path = f"./wikiart_preprocessed_128/{genre}/{name}.pkl"
    
    
    os.makedirs(f"./wikiart_preprocessed_128/{genre}/", exist_ok=True)
    torch.save(tt_tensor, save_path)
