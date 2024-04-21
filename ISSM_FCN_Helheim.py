 ### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm
import time
import pickle

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from torch_geometric.loader import DataLoader
 
# from torch.utils.tensorboard import SummaryWriter

from DGL_model import *
from functions import *

import argparse
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
) -> None:
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch Example')   
    parser.add_argument(
        '--model-dir',
        default='../model',
        help='Model directory',
    )
    parser.add_argument(
        '--log-dir',
        default='./logs/torch_unet',
        help='TensorBoard/checkpoint directory',
    )
    parser.add_argument(
        '--checkpoint-format',
        default='checkpoint_unet_{epoch}.pth.tar',
        help='checkpoint file format',
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='epochs between checkpoints',
    )
    parser.add_argument(
        '--no-cuda',
        # action='store_true',
        default=False,
        help='disables CUDA training',
    )    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        metavar='S',
        help='random seed (default: 42)',
    )
    
    # Training settings
    parser.add_argument(
        '--batch-size',
        type=int,
        default=24,
        metavar='N',
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--phy',
        type=str,
        default='nophy',
        help='filename of dataset',
    )
    parser.add_argument(
        '--in-ch',
        type=int,
        default=10,
        help='Number of input channels',
    )
    parser.add_argument(
        '--hidden-ch',
        type=int,
        default=128,
        help='Number of input channels',
    )
    parser.add_argument(
        '--out-ch',
        type=int,
        default=5,
        help='Number of output channels (6: include all or 3: u, v, h)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='base learning rate (default: 0.01)',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="gcn",
        help='types of the neural network model (e.g. unet, cnn, fc)',
    )
    parser.add_argument(
        '--mesh',
        type=int,
        default=10000,
        help='meshsize of the finite element of ISSM model (select 5000, 10000, or 20000)',
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    
    try:
        # Set automatically by torch distributed launch
        parser.add_argument(
            '--local-rank',
            type=int,
            default=0,
            help='local rank for distributed training',
        )
    except:
        pass
    
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
                

##########################################################################################

import torch.distributed as dist

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import dgl
from dgl.data import DGLDataset

class ISSM_train_dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        glist, _ = load_graphs(self.url)
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
class ISSM_val_dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        glist, _ = load_graphs(self.url)
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
class ISSM_test_dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        glist, _ = load_graphs(self.url)
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
### MAKE INPUT DATASETS #########################################################
class FCN_Dataset(Dataset):
    def __init__(self, input_grid, output_grid):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.input = input_grid
        self.output = output_grid
        
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.output)
    def __getitem__(self, n):

        cnn_input = torch.tensor(self.input[n], dtype=torch.float32)
        cnn_input[torch.isnan(cnn_input)] = 0        
        cnn_output = torch.tensor(self.output[n], dtype=torch.float32)
        cnn_output[torch.isnan(cnn_output)] = 0        
        # cnn_output = torch.transpose(cnn_output, 0, 1)
                        
        return (cnn_input, cnn_output)

def make_sampler_and_loader(args, train_dataset, shuffle = True):
    """Create sampler and dataloader for train and val datasets."""
    torch.set_num_threads(4)
    kwargs: dict[str, Any] = (
        {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    )
    
    if args.cuda:
        kwargs['prefetch_factor'] = 8
        kwargs['persistent_workers'] = True
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **kwargs,
        )
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **kwargs,
        )
    return train_sampler, train_loader

###############################################################################
# Data Loader Preparation
# -----------------------
#
# We split the dataset into training, validation and test subsets. In dataset
# splitting, we need to use a same random seed across processes to ensure a
# same split. We follow the common practice to train with multiple GPUs and
# evaluate with a single GPU, thus only set `use_ddp` to True in the
# :func:`~dgl.dataloading.pytorch.GraphDataLoader` for the training set, where 
# `ddp` stands for :func:`~torch.nn.parallel.DistributedDataParallel`.
#

from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader

def get_dataloaders(dataset, seed, batch_size=32):
    # Use a 80:10:10 train-val-test split
    train_set, val_set, test_set = split_dataset(dataset,
                                                 frac_list=[0.8, 0.15, 0.05],
                                                 shuffle=False,
                                                 random_state=seed)
    train_loader = GraphDataLoader(train_set, use_ddp=True, batch_size=batch_size, shuffle=False)
    val_loader = GraphDataLoader(val_set, batch_size=batch_size, shuffle=False)
    # test_loader = GraphDataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader #, test_loader

###############################################################################
# To ensure same initial model parameters across processes, we need to set the
# same random seed before model initialization. Once we construct a model
# instance, we wrap it with :func:`~torch.nn.parallel.DistributedDataParallel`.
#

import torch
from torch.nn.parallel import DistributedDataParallel

###############################################################################
# Main Function for Each Process
# -----------------------------
#
# Define the model evaluation function as in the single-GPU setting.
#

def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    total_correct = 0

    for bg in dataloader:
        bg = bg.to(device)
        feats = bg.ndata['feat']
        labels = bg.ndata['labels']
        with torch.no_grad():
            pred = model(bg, feats)
        _, pred = torch.max(pred, 1)
        total += len(labels)
        total_correct += (pred == labels).sum().cpu().item()

    return 1.0 * total_correct / total

###############################################################################
# Define the main function for each process.
#

from torch.optim import Adam

def main():    
    
    world_size = int(os.environ['WORLD_SIZE'])
    args = parse_args()
    seed = args.seed
    
    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://',
        world_size=world_size
    )
    
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        
    # for r in range(torch.distributed.get_world_size()):
    #     if r == torch.distributed.get_rank():
    #         print(
    #             f'Global rank {torch.distributed.get_rank()} initialized: '
    #             f'local_rank = {args.local_rank}, '
    #             f'world_size = {torch.distributed.get_world_size()}',
    #         )
    #     torch.distributed.barrier()
    
    model_dir = args.model_dir   

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    torch.cuda.empty_cache()
    
    mesh = args.mesh
    
    train_files, val_files, test_files = generate_list(region = "Helheim", model = "cnn")
    train_dataset = CNN_Helheim_Dataset(train_files[:])
    val_dataset = CNN_Helheim_Dataset(val_files[:])
    test_dataset = CNN_Helheim_Dataset(test_files[:])
    
    # NaN should be 1 (True)
    mask = torch.where(torch.tensor(val_dataset[0][0][1]) > 0, 0, 1)
    
    train_sampler, train_loader = make_sampler_and_loader(args, train_dataset, shuffle = True) 
    val_sampler, val_loader = make_sampler_and_loader(args, val_dataset, shuffle = False)

    n_nodes = 14297 #14517 #23466 #val_dataset[0].num_nodes #val_graphs[0].num_nodes()
    in_channels = args.in_ch #train_dataset[0][0].shape[0] - 2 #val_graphs[0].ndata['feat'].shape[1]-1
    hidden_channels = args.hidden_ch
    if args.out_ch > 0:
        out_channels = args.out_ch
    else:
        out_channels = val_set[0].ndata['label'].shape[1]

    row, col = val_dataset[0].shape[:2]
    if args.local_rank == 0:
        print(f"## IN: {in_channels}; OUT: {out_channels} ({row} x {col})")
        print(f"## Train: {len(train_dataset)}; Val: {len(val_dataset)}; Test: {len(test_dataset)}")
        print("######## TRAINING/VALIDATION DATA IS PREPARED ########")   
    
    # Sampling index for FCN (Convert grid into points) ======================================================
#     xy = train_graphs[0].ndata['feat'][:, 0:2]
#     sampling = torch.zeros(xy.shape, dtype=torch.int)
#     xy_grid = torch.tensor(train_grid_input[0, 0:2], dtype=torch.float32)
#     xy_grid[torch.isnan(xy_grid)] = -1

#     for i in range(0, xy.shape[0]):
#         distance = (xy_grid[0]-xy[i,0])**2 + (xy_grid[1]-xy[i,1])**2
#         k = torch.where(distance == torch.min(distance))
#         sampling[i, 0] = k[0].item()
#         sampling[i, 1] = k[1].item()
    # ==============================================================================
    
    if args.model_type == "cnn":
        model = CNN(in_channels, out_channels, n_nodes, nrow, ncol, hidden_channels)  # convolutional network
    elif args.model_type == "fcn":
        model = FCN(in_channels, out_channels, hidden_channels)
    
    model_name = f"torch_dgl_Helheim_{args.model_type}_{n_nodes}_lr{lr}_in{in_channels}_ch{out_channels}_ft{hidden_channels}"
    print(model_name)
    
    torch.manual_seed(seed)
    
    model.to(device)
    if args.no_cuda:
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    
    criterion = single_loss(mask) #single_loss(mask) #nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)
    scheduler = ExponentialLR(optimizer, gamma=0.98)
    
    total_params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f"MODEL: {args.model_type}; Number of parameters: {total_params}")
    
    history = {'loss': [], 'val_loss': [], 'time': []}
    ti = time.time()
    
    torch.distributed.barrier()
    
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        
        ##### TRAIN ###########################
        train_loss = 0
        train_count = 0
        for (data, target) in train_loader:
            if in_channels == 10:            
                data = data[:, 2:]
            elif in_channels == 7:
                data = data[:, [2,4,5,6,9,10,11]]
            elif in_channels == 8:
                data = data[:, [2,3,4,5,6,9,10,11]]
                
            if out_channels == 4:
                target = target[:, [0,1,4,5], :, :].to(device)
            elif out_channels > 3:
                target = target.to(device)
            elif out_channels == 3:
                target = target[:, [0,1,5], :, :].to(device)
            
            pred = model(data)

            loss = criterion(pred*100, target*100)
            train_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count += 1
        scheduler.step()
        
        ##### VALIDATION ######################
        val_loss = 0
        val_count = 0
        for (data, target) in val_loader:
            if in_channels == 10:            
                data = data[:, 2:]
            elif in_channels == 7:
                data = data[:, [2,4,5,6,9,10,11]]
            elif in_channels == 8:
                data = data[:, [2,3,4,5,6,9,10,11]]
                
            if out_channels == 4:
                target = target[:, [0,1,4,5], :, :].to(device)
            elif out_channels > 3:
                target = target.to(device)
            elif out_channels == 3:
                target = target[:, [0,1,5], :, :].to(device)
            pred = model(data)
            loss = criterion(pred*100, target*100)
            val_loss += loss.cpu().item()
            val_count += 1
            
        history['loss'].append(train_loss/train_count)
        history['val_loss'].append(val_loss/val_count)
        history['time'].append(time.time() - ti)
        
        t1 = time.time() - t0
        if args.local_rank == 0:
            if epoch % 10 == 0:            
                print('Epoch {0} >> Train loss: {1:.4f}; Val loss: {2:.4f} [{3:.2f} sec]'.format(str(epoch).zfill(3), train_loss/train_count, val_loss/val_count, t1))
            if epoch == n_epochs-1:
                print('Epoch {0} >> Train loss: {1:.4f}; Val loss: {2:.4f} [{3:.2f} sec]'.format(str(epoch).zfill(3), train_loss/train_count, val_loss/val_count, t1))
                
                torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
                with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
                    pickle.dump(history, file)
        
#     if args.local_rank == 0:
#         ##### TEST ######################## 
#         rates = np.zeros(len(test_dataset))
#         years = np.zeros(len(test_dataset))

#         if out_channels == 6:
#             scaling = np.array([1, 5000, 5000, 5000, 4000, 3000])
#         elif out_channels == 3:
#             scaling = np.array([5000, 5000, 4000])
            
#         y_pred = np.zeros([len(test_dataset), n_nodes, out_channels])
#         y_true = np.zeros([len(test_dataset), n_nodes, out_channels])

#         x_inputs = np.zeros([len(test_dataset), n_nodes, in_channels])

#         for k, (data, target) in enumerate(DataLoader(test_dataset, batch_size = 1)):
#             data = data[:, :-1]

#             if out_channels > 3:
#                 target = target.to(device)
#             elif out_channels == 3:
#                 target = target[:, [0,1,3], :, :].to(device)
                
#             rates[k] = test_graphs[k].ndata['feat'][0, 2]
#             years[k] = test_graphs[k].ndata['feat'][0, 3] * 20

#             with torch.no_grad():
#                 pred = model(data)
#                 for n in range(0, n_nodes):
#                     y_pred[k, n, :] = pred[:, :out_channels, sampling[n][0], sampling[n][1]].cpu()
#                 # y_pred[k] = pred[0, :, :out_channels].to('cpu')
#                 y_true[k] = test_graphs[k].ndata['label'][:, [0,1,3]].to('cpu')
#                 x_inputs[k] = test_graphs[k].ndata['feat'][:, :-1].to('cpu')

#         test_save = [rates, years, x_inputs, y_true, y_pred]

#         with open(f'../results/test_{model_name}.pkl', 'wb') as file:
#             pickle.dump(test_save, file)
            
#         print("##### Validation done! #####")
        
    dist.destroy_process_group()

###############################################################################
if __name__ == '__main__':
    main()

