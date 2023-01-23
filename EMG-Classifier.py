

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import dataset

from model.mobilenetv2 import MobileNetV2
from dataset import dataset
import torch
import torch.nn as nn
import os
best = 0
training_path = "data/noise/LC"
testing_path = "data/noise/LC"
training_condition = training_path.split("/")[1]
testing_condition = testing_path.split("/")[1]
if len(training_path.split("/")) == 2:
    training_dataset_type = training_path.split("/")[1]
else:
    training_dataset_type = training_path.split("/")[2]
if len(testing_path.split("/")) == 2:
    testing_dataset_type = testing_path.split("/")[1]
else:
    testing_dataset_type = testing_path.split("/")[2]

network_type = "Mobilenet"
base_log_path = f"Experiment_data/ICELab/{network_type}"
base_weight_path = f"weight/ICELab/{network_type}"
weight_path = f"{base_weight_path}/Training_{training_condition}_test{testing_condition}/{training_dataset_type}_{testing_dataset_type}"
log_path = f"{base_log_path}/Training_{training_condition}_test{testing_condition}/{training_dataset_type}_{testing_dataset_type}"

training_dataset = dataset(root=training_path,image_size1=8,image_size2=24,train=True)
testing_dataset = dataset(root=testing_path,image_size1=8,image_size2=24,train=False)

training_dataloader = DataLoader(training_dataset,batch_size=1000,shuffle=True,num_workers=16)
testing_dataloader = DataLoader(testing_dataset,batch_size=1000,shuffle=False,num_workers=16)

model = MobileNetV2(num_classes=8,input_layer=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
best_acc = 0
def train(epoches,model,optimizer,criterion,dataloader):

    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    with tqdm(total=len(dataloader)) as pbar:
        for i, data in enumerate(dataloader, 0):
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
    return round(running_loss/len(dataloader),4), round(100 * correct / total,4)
    

def test(epoch,model,criterion,dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients

                # forward + backward + optimize
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                pbar.update()
                pbar.set_description(f"Epoch: {epoch} | Loss: {running_loss/(i+1):.4f} | ACC: {100 * correct / total:.4f}%")
    
    print('Finished validation')
    return round(running_loss/len(dataloader),4), round(100 * correct / total,4)


epochs = 50
epoch = 0
while epoch < epochs:
    train_loss,train_acc = train(epoch,model,optimizer,criterion,training_dataloader)
    test_loss,test_acc = test(epoch,model,criterion,testing_dataloader)
    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)
    if test_acc > best:
        best = test_acc
        torch.save(model.state_dict(),f"{weight_path}/{best}")
        
    scheduler.step()
    epoch+=1
