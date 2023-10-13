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
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torch_geometric.loader import DataLoader
 
# from torch.utils.tensorboard import SummaryWriter

from DGL_model import *

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
        default=8,
        metavar='N',
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--batches-per-allreduce',
        type=int,
        default=1,
        help='number of batches processed locally before '
        'executing allreduce across workers; it multiplies '
        'total batch size.',
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=8,
        help='input batch size for validation (default: 16)',
    )
    parser.add_argument(
        '--phy',
        type=str,
        default='nophy',
        help='filename of dataset',
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
        default="egcn",
        help='types of the neural network model (e.g. unet, cnn, fc)',
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    
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
    def __init__(self):
        super().__init__(name='pig')
        
    def process(self):
        glist, _ = load_graphs("../data/DGL_train_dataset.bin")
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
class ISSM_test_dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='pig')
        
    def process(self):
        glist, _ = load_graphs("../data/DGL_test_dataset.bin")
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)

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
                                                 shuffle=True,
                                                 random_state=seed)
    train_loader = GraphDataLoader(train_set, use_ddp=True, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=batch_size)
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
    
    seed = 0
    world_size = int(os.environ['WORLD_SIZE'])
    args = parse_args()
    
    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://'
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
    val_batch = args.val_batch_size  # size of validation batch size
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        
    print("######## TRAINING/VALIDATION DATA IS PREPARED ########")
    
    torch.cuda.empty_cache()
    
    n_nodes = 1112
    in_channels = 4
    out_channels = 3
    
    if args.model_type == "gcn":
        model = GCN(in_channels, out_channels, 128)  # Graph convolutional network    
    elif args.model_type == "gin":
        model = GIN(in_channels, out_channels, 128)  # Equivariant Graph convolutional network
    elif args.model_type == "mlp":
        model = MLP(in_channels, out_channels, 128)  # Fully connected network
    elif args.model_type == "egcn":
        model = EGNNConv(in_channels, 128, out_channels, 1) # Equivariant Graph convolutional network
    
    model_name = f"torch_dgl_{args.model_type}_lr{lr}_{phy}_ch{out_channels}"
    
    torch.manual_seed(seed)
    
    model.to(device)
    if args.no_cuda:
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)    
    
    train_set = ISSM_train_dataset()
    train_loader, val_loader = get_dataloaders(train_set, seed, batch_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        train_loader.set_epoch(epoch)
        
        ##### TRAIN ###########################
        train_loss = 0
        train_count = 0
        for bg in train_loader:
            bg = bg.to(device)
            feats = bg.ndata['feat']
            coord_feat = bg.ndata['feat'][:, :2]
            edge_feat = bg.edata['weight'].float() #.repeat(1, 2)
            if out_channels == 6:
                labels = bg.ndata['label']
            elif out_channels == 3:
                labels = bg.ndata['label'][:, [1,2,4]]
            if args.model_type == "egcn":
                pred = model(bg, feats, coord_feat, edge_feat)
                labels = torch.cat([labels, coord_feat], dim=1)
            else:
                pred = model(bg, feats)

            loss = criterion(pred*100, labels*100)
            train_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count += 1   
        
        ##### VALIDATION ######################
        val_loss = 0
        val_count = 0
        for bg in val_loader:
            bg = bg.to(device)
            feats = bg.ndata['feat']
            coord_feat = bg.ndata['feat'][:, :2]
            edge_feat = bg.edata['weight'].float() #.repeat(1, 2)
            if out_channels == 6:
                labels = bg.ndata['label']
            elif out_channels == 3:
                labels = bg.ndata['label'][:, [1,2,4]]
            
            with torch.no_grad():
                if args.model_type == "egcn":
                    pred = model(bg, feats, coord_feat, edge_feat)
                    labels = torch.cat([labels, coord_feat], dim=1)
                else:
                    pred = model(bg, feats)
            loss = criterion(pred*100, labels*100)
            val_loss += loss.cpu().item()
            val_count += 1
        
        t1 = time.time() - t0
        if epoch % 5 == 0 or epoch == n_epochs-1:
            if args.local_rank == 0:
                print('Epoch {0} >> Train loss: {1:.4f}; Val loss: {2:.4f} [{3:.2f} sec]'.format(str(epoch).zfill(3), train_loss/train_count, val_loss/val_count, t1))
        
    if args.local_rank == 0:
        test_set = ISSM_test_dataset()
        ##### TEST ########################
        rates = np.zeros(len(test_set))
        years = np.zeros(len(test_set))

        if out_channels == 6:
            scaling = [1, 5000, 5000, 5000, 4000]
        elif out_channels == 3:
            scaling = [5000, 5000, 4000]
        y_pred = np.zeros([len(test_set), n_nodes, out_channels])
        y_true = np.zeros([len(test_set), n_nodes, out_channels])
        x_inputs = np.zeros([len(test_set), n_nodes, in_channels])

        for k, bg in enumerate(test_set):
            bg = bg.to(device)
            feats = bg.ndata['feat']
            coord_feat = bg.ndata['feat'][:, :2]
            edge_feat = bg.edata['weight'].float() #.repeat(1, 2)
            if out_channels == 6:
                labels = bg.ndata['label']
            elif out_channels == 3:
                labels = bg.ndata['label'][:, [1,2,4]]
                
            rates[k] = r
            years[k] = year

            with torch.no_grad():
                if args.model_type == "egcn":
                    pred = model(bg, feats, coord_feat, edge_feat)
                    labels = torch.cat([labels, coord_feat], dim=1)
                else:
                    pred = model(bg, feats)
                y_pred[k] = pred[:, :out_channels].to('cpu')
                y_true[k] = labels[:, :out_channels].to('cpu')
                x_inputs[k] = feats.to('cpu')

        test_save = [rates, years, x_inputs, y_true, y_pred]

        with open(f'../results/test_{model_name}.pkl', 'wb') as file:
            pickle.dump(test_save, file)
            
        print("##### Validation done! #####")
    dist.destroy_process_group()

###############################################################################
if __name__ == '__main__':
    main()

