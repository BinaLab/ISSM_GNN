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
class CNN_Dataset(Dataset):
    def __init__(self, input_grid, output_graph):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.input = input_grid
        self.output = output_graph
        self.length = len(self.valid)
        
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.valid)
    def __getitem__(self, n):
        
        _, var_ip, row, col = self.input.shape
        _, var_op, _, _ = self.output.shape

        cnn_input = torch.tensor(self.input[n], dtype=torch.float32)
        cnn_input[torch.isnan(cnn_input)] = 0
        
        cnn_output = self.output[n].ndata['label']
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
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    torch.cuda.empty_cache()
    
    data_ver = data_file[-6:-4]
    data_type = data_file[6:9]
    
    mesh = args.mesh
    train_graphs = load_graphs(f"../data/DGL_train_dataset_g{mesh}.bin")
    val_graphs = load_graphs(f"../data/DGL_val_dataset_g{mesh}.bin")
    test_graphs = load_graphs(f"../data/DGL_test_dataset_g{mesh}.bin")
    
    with open(f'../data/CNN_dataset_g{mesh}.pkl', 'rb') as file:
        [grid_input, _, train_index] = pickle.load(file)
    
    _, _, nrow, ncol = grid_input.shape
    train_grid = grid_input[train_index == 0]
    val_grid = grid_input[train_index == 1]
    test_grid = grid_input[train_index == 2]
    
    train_dataset = CNN_Dataset(train_grid, train_graphs)
    val_dataset = CNN_Dataset(val_grid, val_graphs)
    test_dataset = CNN_Dataset(test_grid, test_graphs)
    
    train_sampler, train_loader = make_sampler_and_loader(args, train_dataset, shuffle = True) 
    val_sampler, val_loader = make_sampler_and_loader(args, val_dataset, shuffle = False)

    n_nodes = val_grid[0].num_nodes()
    in_channels = val_grid[0].ndata['feat'].shape[1]-1
    
    if args.out_ch == 3:
        out_channels = args.out_ch
    else:
        out_channels = val_set[0].ndata['label'].shape[1]
    
    if args.local_rank == 0:
        print(f"## NODE: {n_nodes}; IN: {in_channels}; OUT: {out_channels}")
        print(f"## Train: {len(train_set)}; Val: {len(val_set)}; Test: {len(test_set)}")
        print("######## TRAINING/VALIDATION DATA IS PREPARED ########")   
    
    if args.model_type != "cnn":
        args.moel_type = "cnn"
        
    model = CNN(in_channels, out_channels, n_nodes, nrow, ncol, 128)  # Graph convolutional network

    model_name = f"torch_dgl_{args.model_type}_{n_nodes}_lr{lr}_{phy}_ch{out_channels}"
    
    torch.manual_seed(seed)
    
    model.to(device)
    if args.no_cuda:
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)
    scheduler = ExponentialLR(optimizer, gamma=0.98)
    
    total_params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f"MODEL: {args.model_type}; Number of parameters: {total_params}")
    
    history = {'loss': [], 'val_loss': [], 'time': []}
    ti = time.time()
    
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        train_loader.set_epoch(epoch)
        
        ##### TRAIN ###########################
        train_loss = 0
        train_count = 0
        for (data, target) in train_loader:
            data = data[:, :-1]            
            if out_channels > 3:
                pass
            elif out_channels == 3:
                target = target[:, [0,1,3]]
            
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
            data = data[:, :-1]            
            if out_channels > 3:
                pass
            elif out_channels == 3:
                target = target[:, [0,1,3]]            
            pred = model(data)
            loss = criterion(pred*100, labels*100)
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
        
    if args.local_rank == 0:
        ##### TEST ########################
        rates = np.zeros(len(test_dataset))
        years = np.zeros(len(test_dataset))

        if out_channels == 6:
            scaling = np.array([1, 5000, 5000, 5000, 4000, 3000])
        elif out_channels == 3:
            scaling = np.array([5000, 5000, 4000])
            
        y_pred = np.zeros([len(test_dataset), n_nodes, out_channels])
        y_true = np.zeros([len(test_dataset), n_nodes, out_channels])

        x_inputs = np.zeros([len(test_dataset), n_nodes, in_channels])

        for k, (data, target) in enumerate(test_dataset):
            data = data[:, :-1]            
            if out_channels > 3:
                pass
            elif out_channels == 3:
                target = target[:, [0,1,3]]
                
            rates[k] = feats[0, 2]
            years[k] = feats[0, 3] * 20

            with torch.no_grad():
                pred = model(data)
                y_pred[k] = pred[:, :out_channels].to('cpu')
                y_true[k] = labels[:, :out_channels].to('cpu')
                x_inputs[k] = test_graphs[k].ndata['feat'].to('cpu')

        test_save = [rates, years, x_inputs, y_true, y_pred]

        with open(f'../results/test_{model_name}.pkl', 'wb') as file:
            pickle.dump(test_save, file)
            
        print("##### Validation done! #####")
    dist.destroy_process_group()

###############################################################################
if __name__ == '__main__':
    main()

