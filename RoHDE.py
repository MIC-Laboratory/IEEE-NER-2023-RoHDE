
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import dataset
from model.Generator import Generator
from model.mobilenetv2 import MobileNetV2
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import time



TESTING_PATH = "data/clean"
GAN_WEIGHT = "weight/Gan/CA"
MOBILENET_WEIGHT = "weight/ICELab/Mobilenet/Training_clean_testclean/clean_clean/98.0762"
IMAGE_SIZE1 = 8
IMAGE_SIZE2 = 24
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
EPOCH = 0
FEATURES_GEN = 16
EMBBED_SIZE = 100
NUM_CLASSES = 8
BATCH_SIZE = 512
BASE_LOG_PATH = "Experiment_data/GAN"
TESTING_DATASET_TYPE = TESTING_PATH.split("/")[-1]
VOTING_WINDOW = 5
VOTING_WINDOW_LIST = [1,2,3,4,5,6,10]
LOG_PATH = f"{BASE_LOG_PATH}/{TESTING_DATASET_TYPE}"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
testing_dataset = dataset(TESTING_PATH,IMAGE_SIZE1,IMAGE_SIZE2,train=False)

testing_dataloader = DataLoader(testing_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=16)

model = MobileNetV2(num_classes=8,input_layer=1)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,IMAGE_SIZE1,IMAGE_SIZE2,EMBBED_SIZE,NUM_CLASSES).to(device)
gen.load_state_dict(torch.load(GAN_WEIGHT))
model.load_state_dict(torch.load(MOBILENET_WEIGHT))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
gen.to(device)
gen.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
writer = SummaryWriter(LOG_PATH)



def train(epoches,model,optimizer,criterion):

    model.train()
    gen.eval()

    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
    labels = torch.randint(0,8,(BATCH_SIZE,)).to(device)
    fake = gen(fixed_noise,labels)
    sample = (fake,labels)
    correct = 0
    total = 0
    running_loss = 0.0
    
   
    with tqdm(total=1) as pbar:
        for i, data in enumerate([sample], 0):
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            pbar.update()
            pbar.set_description(f"Epoch: {epoches} | Loss: {running_loss/(i+1):.4f} | ACC: {100 * correct / total:.4f}%")
    print('Finished Training')
    return round(running_loss,4), round(100 * correct / total,4)
    

def test(epoch,model,dataloader,voting):
    model.eval()
    correct = 0
    total = 0
    voting = voting
    voting_count = 0
    voting_output = []
    laterncy = 0
    with tqdm(total=len(dataloader)//voting) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                start = time.time()
                if voting_count < voting:
                
                    inputs, labels = data
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    _, predicted = torch.max(outputs, 1)
                    voting_output.append(predicted)
                    voting_count+=1
                    if voting_count < voting:
                        continue
                
                
 
                predicted = torch.stack(voting_output)
                predicted = torch.mode(predicted,0).values
                voting_output = []
                voting_count = 0
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                end = time.time()
                pbar.update()
                pbar.set_description(f"Epoch: {epoch} | ACC: {100 * correct / total:.4f}%")
                laterncy += (end-start)
    print('Finished validation')
    return round(100 * correct / total,4), round(laterncy/(len(dataloader)//voting),4)


temp_vote_list = []
laterncy_list = []
while EPOCH < NUM_EPOCHS:

    train_loss,train_acc = train(EPOCH,model,optimizer,criterion)
    
    test_acc, laterncy = test(EPOCH,model,testing_dataloader,VOTING_WINDOW)

    writer.add_scalar("ACC V.S. Samples (VOTING WINDOWS = 5)",test_acc,EPOCH*BATCH_SIZE)
    scheduler.step()
    EPOCH+=1

for vote in VOTING_WINDOW_LIST:
    EPOCH = 0
    model.load_state_dict(torch.load(MOBILENET_WEIGHT))
    while EPOCH < NUM_EPOCHS:
        train_loss,train_acc = train(EPOCH,model,optimizer,criterion)
        scheduler.step()
        EPOCH+=1
    test_acc, laterncy = test(0,model,testing_dataloader,vote)
    writer.add_scalar(f"ACC V.S. Latency (Sample = {NUM_EPOCHS*BATCH_SIZE})",test_acc,vote)
    laterncy_list.append(laterncy)
    temp_vote_list.append(test_acc)

print("Vote acc:",temp_vote_list)
print("laterncy:",laterncy_list)

