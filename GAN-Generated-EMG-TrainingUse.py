from genericpath import isfile
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import dataset
from model.Generator import Generator
from model.mobilenetv2 import MobileNetV2
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
for i in tqdm(range(1,65),position=0):
    if i == 10:
        continue
    
    GAN_WEIGHT = f"/home/zhenyulin/sEMG_2D_Conv/Gesture_Recognition_using_a_CNN/weight/TestGan/CA_Channel{i}"
    IMAGE_SIZE1 = 3840
    CHANNELS_IMG = 1
    Z_DIM = 100
    NUM_EPOCHS = 100
    EPOCH = 0
    FEATURES_GEN = 16
    EMBBED_SIZE = 100
    NUM_CLASSES = 8
    BATCH_SIZE = 512
    for j in tqdm(range(25),position=2,leave=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,IMAGE_SIZE1,EMBBED_SIZE,NUM_CLASSES).to(device)
        gen.load_state_dict(torch.load(GAN_WEIGHT))
        gen.to(device)
        gen.eval()
        x =torch.linspace(-torch.pi,torch.pi,Z_DIM)
        fixed_noise = (0.1 * torch.randn(BATCH_SIZE, Z_DIM)+0.9*torch.sin(x))
        fixed_noise = fixed_noise.unsqueeze(2).to(device)
        labels = torch.randint(0,8,(BATCH_SIZE,)).to(device)
        fake = gen(fixed_noise,labels)
        labels = labels.cpu().detach().numpy()
        fake_np = fake.cpu().detach().numpy()
        fake_np = fake_np.reshape(BATCH_SIZE,-1)
        if not os.path.isdir(f"Generated CA/Channel_{i}"):
            os.mkdir(f"Generated CA/Channel_{i}")
        np.save(f"Generated CA/Channel_{i}/{j}_data.npy", fake_np)
        np.save(f"Generated CA/Channel_{i}/{j}_label.npy", labels)
    
    
    