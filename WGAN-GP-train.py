
"""
Training of WGAN-GP
"""

import torch
import torch.optim as optim
import os
import PIL.Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.util import gradient_penalty, initialize_weights, cross_correlation,gen_plot,loss_fft
from model.Discriminator import Discriminator
from model.Generator import Generator
from dataset import dataset
from tqdm import tqdm
# from model.mobilenetv2 import MobileNetV2
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from torchvision.transforms import ToTensor
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE1 = 8
IMAGE_SIZE2 = 24
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 3000
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

EMBBED_SIZE = 100
NUM_CLASSES = 8
ROOT = "data/noise/CA"
NUM_WORKERS = 2
# MOBILENET_PRETRAIN_WEIGHT = "weight/ICELab/Mobilenet/Training_noise_testnoise/CA_CA/94.2069"
BASE_WEIGHT_PATH = "weight/TestGan/"
BASE_LOG_PATH = "Experiment_data/TestGAN/"
DATASET_TYPE = ROOT.split("/")[-1]

for i in range(1):
    GAN_SAVE_WEIGHT = BASE_WEIGHT_PATH+DATASET_TYPE


    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    datasets = dataset(root=ROOT,image_size1=IMAGE_SIZE1,image_size2=IMAGE_SIZE2, all=True,train=True,channel=i,generated_CA_root="Generated CA")
    loader = DataLoader(
        datasets,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    writer = SummaryWriter(BASE_LOG_PATH)


    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,IMAGE_SIZE1,IMAGE_SIZE2,EMBBED_SIZE,NUM_CLASSES).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC,IMAGE_SIZE1,IMAGE_SIZE2,NUM_CLASSES).to(device)
    # mobilenet = MobileNetV2(num_classes=8,input_layer=1).to(device=device)
    initialize_weights(gen)
    initialize_weights(critic)
    # mobilenet.load_state_dict(torch.load(MOBILENET_PRETRAIN_WEIGHT))

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    step = 0

    gen.train()
    critic.train()
    temp_batch_idx = 0
    x =torch.linspace(-torch.pi,torch.pi,Z_DIM).to(device)
    # mobilenet.eval()
    for epoch in range(NUM_EPOCHS):
        correlation = 0
        last_label = None
        last_real = None
        last_noise = None
        with tqdm(total=len(loader)) as pbar:
            for batch_idx, (real, labels) in enumerate(loader):
                real = real.to(device)
                labels = labels.to(device)
                cur_batch_size = real.shape[0]

                for _ in range(CRITIC_ITERATIONS):
                    
                    noise = (0.1 * torch.randn(cur_batch_size, Z_DIM,device=device)+0.9*torch.sin(x).to(device))
                    noise = noise.unsqueeze(2).unsqueeze(3).to(device)
                    fake = gen(noise,labels)
                    critic_real = critic(real,labels).reshape(-1)
                    critic_fake = critic(fake,labels).reshape(-1)
                    gp = gradient_penalty(critic, labels, real, fake, device=device)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                    )
                    critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    opt_critic.step()

                gen_fake = critic(fake,labels).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                
                
                pbar.update()
                pbar.set_description(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )
                if not os.path.isdir(BASE_WEIGHT_PATH):
                    os.makedirs(BASE_WEIGHT_PATH)
                torch.save(gen.state_dict(),GAN_SAVE_WEIGHT)
                
                last_label = labels
                last_real = real
                last_noise = noise
                last_best_index = torch.argmax(gen_fake)
            with torch.no_grad():
                fake = gen(last_noise,last_label)
                correlation = cross_correlation((last_real.detach().cpu().numpy().reshape(-1)),(fake.detach().cpu().numpy().reshape(-1)))
                dtw,path = fastdtw((last_real.detach().cpu().numpy().reshape(-1)),(fake.detach().cpu().numpy().reshape(-1)),dist=2)
            writer.add_scalar("Training D Loss",round(loss_critic.item(),4),epoch)
            writer.add_scalar("Training G Loss",round(loss_gen.item(),4),epoch)
            writer.add_scalar("correlation",round(correlation.item(),4),epoch)
            writer.add_scalar("DTW",round(dtw,4),epoch)
            
            fake_plot_buf = gen_plot(fake.cpu().detach().numpy().reshape(-1),"fake EMG")
            real_plot_buf = gen_plot(last_real.cpu().detach().numpy().reshape(-1),"read EMG")
            fake_image = PIL.Image.open(fake_plot_buf)
            real_image = PIL.Image.open(real_plot_buf)
            fake_image = ToTensor()(fake_image)
            real_image = ToTensor()(real_image)


            writer.add_image('fake EMG', fake_image, epoch)
            writer.add_image('Real EMG', real_image, epoch)