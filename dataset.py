from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from torchvision.transforms.functional import normalize
import torch
class dataset(Dataset):

    def __init__(self, root,image_size1,image_size2=1,generated_CA_root="",train=False,all=False,channel=1):
        self.root = root
        self.channel = channel
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.data,self.label = self.extra_data(root,train,all)

        

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        

        data = self.data[idx]
        label = self.label[idx].astype(int)
        data = self.NormalizeData(data)

        
        data = data.reshape(1,self.image_size1,self.image_size2)

        data = data.astype("float32")
        return data,label

        
    
    def NormalizeData(self,data):

        return data

    def load_from_numpy(self,root,data,label):
        total_noise_data = None
        total_noise_label = None
        for i in range(1,65):
            if i == 10:
                continue
            noise_data = None
            noise_label = None
            for j in range(25):
                
                data_path = f"{root}/Channel_{i}/{j}_data.npy"
                label_path = f"{root}/Channel_{i}/{j}_label.npy"
                if noise_data is None:
                    noise_data = np.load(data_path).reshape(-1)
                    noise_label = np.load(label_path)
                else:
                    noise_data = np.concatenate((noise_data,np.load(data_path).reshape(-1)))
                    noise_label = np.concatenate((noise_label,np.load(label_path)))
            if total_noise_data is None:
                total_noise_data = noise_data[np.newaxis,:]
                total_noise_label = noise_label[np.newaxis,:]
            else:
                total_noise_data = np.concatenate((total_noise_data,noise_data[np.newaxis,:]),axis=0)
                total_noise_label = np.concatenate((total_noise_label,noise_label[np.newaxis,:]),axis=0)
        original_clean_data = data.copy()
        original_clean_label = label.copy()
        final_data = np.hsplit(total_noise_data,total_noise_data.shape[1])
        final_label = noise_label.repeat(2048)
        for k in range(data.shape[0]):
            original_clean_data[k][126:136] = final_data[k][0:10].reshape(-1)
            original_clean_data[k][192-53:] = final_data[k][63-53:].reshape(-1)
        
        
        
        return original_clean_data,original_clean_label
    def data_filter(self,data,root,label):
        if (root.split("/")[-1] == "LC"):
            filtered_data = data.T
            filtered_data = filtered_data[71]
            filtered_label = None
            for i in range(8):
                if filtered_label is None:
                    filtered_label = torch.Tensor([i]).repeat(label.shape[0]//2048//8)
                else:
                    filtered_label = torch.cat((filtered_label,torch.Tensor([i]).repeat(label.shape[0]//2048//8)))
            filtered_label = filtered_label.repeat(filtered_data.shape[0])
            filtered_data = filtered_data.reshape(-1)
            filtered_data = filtered_data.reshape(-1,2048)
            return filtered_data,filtered_label
        if (root.split("/")[-1] == "CA"):
            max_index = set()
            for i in range(127):
                max_index.add(i)
            filtered_data = data.T
            filtered_data = filtered_data[list(max_index)]
            filtered_data = filtered_data[self.channel]
            filtered_label = None
            for i in range(8):
                if filtered_label is None:
                    filtered_label = torch.Tensor([i]).repeat(label.shape[0]//2048//8)
                else:
                    filtered_label = torch.cat((filtered_label,torch.Tensor([i]).repeat(label.shape[0]//2048//8)))
            filtered_label = filtered_label.repeat(filtered_data.shape[0])
            filtered_data = filtered_data.reshape(-1)
            filtered_data = filtered_data.reshape(-1,2048)
            
            return filtered_data,filtered_label

    def load_from_mat(self,fileAddress):
        return scipy.io.loadmat(fileAddress)['Data']

    def add_data(self,data,labels):
        data = data.reshape(-1,192)
        self.data = np.concatenate([data,self.data],axis=0)
        self.label = np.concatenate([labels,self.label],axis=0)

    def extra_data(self,fileAddress,train=False,all=False):
        data = None
        label = None
        if all:
            start = 1
            end = 6
        else:
            if train:
                start = 1
                end = 5
            else:
                start = 5
                end = 6
        for gest in range(1,9):
            for i in range(start,end):
                if data is None:
                    data = self.load_from_mat(f"{fileAddress}/001-00{gest}-00{i}.mat")

                    label = np.repeat(gest-1,data.shape[0])
                else:
                    temp = self.load_from_mat(f"{fileAddress}/001-00{gest}-00{i}.mat")
        
                    if temp.shape[1] == 193:
                        temp = temp[:,:-1]
                    
                    data = np.concatenate((data,temp),axis=0)
                    label = np.concatenate((label,np.repeat(gest-1,temp.shape[0])),axis=0)
        return data,label