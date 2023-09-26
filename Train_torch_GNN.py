### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
import pickle
import time

# check pytorch version
import torch    
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from functions import *
from torch_model import *

import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

##########################################################################################

data_path = "D:\\PINN\\data\\"

year = 2019
n_epochs = 50
batch_size = 16  # size of each batch
lr = 0.005

loader = torch.load(f'{data_path}/Graph/Grid_graph_{year}_train_batch_{batch_size}.pt')
val_loader = torch.load(f'{data_path}/Graph/Grid_graph_{year}_val_batch_{batch_size}.pt')

print("######## TRAINING/VALIDATION DATA IS PREPARED ########")

net = GCNet()

torch.cuda.empty_cache()
device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# net = nn.DataParallel(net, device_ids=[0])

net.to(device)

##### PHYSICS OR NOT #####
phy = "nophy"
##########################

if device == "cpu":
    device_name = "cpu"
else:
    device_name = "gpu"

model_name = f"torch_gcn_lr{lr}_wo{year}_{phy}_{device_name}"

if phy == "phy":
    loss_fn = physics_loss() # nn.L1Loss() #nn.CrossEntropyLoss()
elif phy == "nophy":
    loss_fn = custom_loss() # nn.L1Loss() #nn.CrossEntropyLoss()
    
optimizer = optim.Adam(net.parameters(), lr=lr)

history = {'loss': [], 'val_loss': [], 'time': []}

total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

## Train model #############################################################

t0 = time.time()
for epoch in range(n_epochs):
    
    net.train()
    train_loss = 0.0
    val_loss = 0.0

    with tqdm(loader, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Train Epoch {epoch}/{n_epochs}")
        for data in bar:
            optimizer.zero_grad()  # Clear gradients.
            out = net(data.x.to(torch.float).to(device), data.edge_index.to(device))  # Perform a single forward pass.
            y_pred = torch.reshape(out, (int(len(data.x)/(320*320)), 320, 320, 3))
            y_true = torch.reshape(data.y, (int(len(data.x)/(320*320)), 320, 320, 3))
            loss = loss_fn(y_pred.to(device), y_true.to(device))  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.  
            bar.set_postfix(loss=float(loss))

            train_loss += loss.item()
            
    net.eval()
    
    with tqdm(val_loader, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Val Epoch {epoch}/{n_epochs}")
        for val_data in bar:
            out = net(val_data.x.to(torch.float).to(device), val_data.edge_index.to(device))  # Perform a single forward pass.
            y_pred = torch.reshape(out, (int(len(val_data.x)/(320*320)), 320, 320, 3))
            y_true = torch.reshape(data.y, (int(len(val_data.x)/(320*320)), 320, 320, 3))
            val_loss += loss_fn(y_pred.to(device), y_true.to(device)).item()  # Compute the loss solely based on the training nodes.
            bar.set_postfix(loss=float(val_loss))
        
    history['loss'].append(train_loss/len(loader))
    history['val_loss'].append(val_loss/len(val_loader))
    history['time'].append(time.time()-t0)

    
torch.save(net.state_dict(), f'../model/{model_name}.pth')

with open(f'../model/history_{model_name}.pkl', 'wb') as file:
    pickle.dump(history, file)


# print(history)

