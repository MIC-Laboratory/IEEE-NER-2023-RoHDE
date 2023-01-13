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

GAN_WEIGHT = "/home/zhenyulin/sEMG_2D_Conv/Gesture_Recognition_using_a_CNN/weight/TestGan/LC_Channel72 (Figure Use)"
IMAGE_SIZE1 = 3840
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
EPOCH = 0
FEATURES_GEN = 16
EMBBED_SIZE = 100
NUM_CLASSES = 8
BATCH_SIZE = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,IMAGE_SIZE1,EMBBED_SIZE,NUM_CLASSES).to(device)
gen.load_state_dict(torch.load(GAN_WEIGHT))
gen.to(device)
gen.eval()
x =torch.linspace(-torch.pi,torch.pi,Z_DIM)
fixed_noise = (0.1 * torch.randn(BATCH_SIZE, Z_DIM)+0.9*torch.sin(x))
fixed_noise = fixed_noise.unsqueeze(2).to(device)
labels = torch.tensor([0]).repeat(BATCH_SIZE).to(device)
# labels = torch.randint(0,1,(BATCH_SIZE,)).to(device)
fake = gen(fixed_noise,labels)
# fake = fake.reshape(-1,192)
print(torch.min(fake))
print(torch.max(fake))
fake_np = fake.cpu().detach().numpy()
fake_np = fake_np.reshape(-1)
# fake_np = (fake_np-np.min(fake_np))/(np.max(fake_np)-np.min(fake_np))

plt.plot(fake_np)
plt.show()
# fake_np = fake_np.reshape(30720,2048)
x_df = pd.DataFrame(fake_np)
x_df.to_csv('GAN-Generated-HD-EMG-LC.csv')
# sample = (fake,labels)



# for i in range(0,100,10):
    
#     ax = sns.heatmap(fake_np[i], linewidth=0.5)
#     plt.show()

# harvest = fake_np[2]


# fig, ax = plt.subplots()
# im = ax.imshow(harvest, cmap='hot', interpolation='nearest')

# Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(farmers)), labels=farmers)
# ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

# Rotate the tick labels and set their alignment.


# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")


# plt.show()