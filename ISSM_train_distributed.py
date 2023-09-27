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
 
# from torch.utils.tensorboard import SummaryWriter

from torch_model import *

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
        default=16,
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
        default=16,
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

def make_sampler_and_loader(args, train_dataset):
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

    
class Metric:
    """Metric tracking class."""

    def __init__(self, name: str):
        """Init Metric."""
        self.name = name
        self.total = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """Update metric.

        Args:
            val (float): new value to add.
            n (int): weight of new value.
        """
        dist.all_reduce(val, async_op=False)
        self.total += val.cpu() / dist.get_world_size()
        self.n += n

    @property
    def avg(self) -> torch.Tensor:
        """Get average of metric."""
        return self.total / self.n
    
def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_sampler: torch.utils.data.distributed.DistributedSampler,
    args
):
    
    """Train model."""
    model.train()
    train_sampler.set_epoch(epoch)
    
    mini_step = 0
    step_loss = torch.tensor(0.0).to('cuda' if args.cuda else 'cpu')
    train_loss = Metric('train_loss')
    t0 = time.time()
    
    with tqdm(
        total=math.ceil(len(train_loader) / args.batches_per_allreduce),
        bar_format='{l_bar}{bar:10}{r_bar}',
        desc=f'Epoch {epoch:3d}/{args.epochs:3d}',
        disable=not args.verbose,
    ) as t:
        for batch_idx, data in enumerate(train_loader):
            print(data)
            mini_step += 1
            if args.cuda:
                data = data.cuda()
                
            y_pred = model(torch.tensor(data.x, dtype=torch.float32), data.edge_index)  # Perform a single forward pass.
            y_true = torch.tensor(data.y, dtype=torch.float32)

            loss = loss_func(y_pred, y_true)

            with torch.no_grad():
                step_loss += loss

            loss = loss / args.batches_per_allreduce

            if (
                mini_step % args.batches_per_allreduce == 0
                or batch_idx + 1 == len(train_loader)
            ):
                loss.backward()
            else:
                with model.no_sync():  # type: ignore
                    loss.backward()

            if (
                mini_step % args.batches_per_allreduce == 0
                or batch_idx + 1 == len(train_loader)
            ):

                optimizer.step()
                optimizer.zero_grad()
                
                train_loss.update(step_loss / mini_step)
                step_loss.zero_()

                t.set_postfix_str('loss: {:.4f}'.format(train_loss.avg))
                t.update(1)
                mini_step = 0

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        
    return train_loss.avg


def validate(
    epoch: int,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    args
):
    """Test the model."""
    model.eval()
    val_loss = Metric('val_loss')

    with tqdm(
        total=len(val_loader),
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='             ',
        disable=not args.verbose
    ) as t:
        with torch.no_grad():
            for i, data in enumerate(val_loader):

                if args.cuda:
                    data = data.cuda()

                y_pred = model(torch.tensor(data.x, dtype=torch.float32), data.edge_index)  # Perform a single forward pass.
                y_true = torch.tensor(data.y, dtype=torch.float32) 
                
                val_loss.update(loss_func(y_pred, y_true))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str(
                        'val_loss: {:.4f}'.format(val_loss.avg),
                        refresh=False,
                    )

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        
    return val_loss.avg

def test(
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    model_name,
    args
):
    """Test the model."""
    model.eval()
    val_loss = Metric('val_loss')

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            val_loss.update(loss_func(output, target))

            test_save = [data.to('cpu').detach().numpy(), target.to('cpu').detach().numpy(), output.to('cpu').detach().numpy()]

            # Open a file and use dump()
            with open(f'../results/test_{model_name}_{args.global_rank}_{i}.pkl', 'wb') as file:
                pickle.dump(test_save, file)
    
##########################################################################################

def main() -> None:    
    
    #### SETTING CUDA ENVIRONMENTS ####
    """Main train and eval function."""
    args = parse_args()

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://',
    )

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    
    if args.no_cuda:
        device = torch.device('cpu')
        device_name = 'cpu'
    else:
        device = torch.device('cuda')
        device_name = 'gpu'
        
    torch.cuda.empty_cache()
    
    args.verbose = dist.get_rank() == 0
    world_size = int(os.environ['WORLD_SIZE'])

    if args.verbose:
        print('Collecting env info...')
        # print(collect_env.get_pretty_env_info())
        # print()

    for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
        torch.distributed.barrier()
    
    args.global_rank = torch.distributed.get_rank()

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    # args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None  
    args.log_writer = None if args.verbose else None  

    model_dir = args.model_dir   

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    val_batch = args.val_batch_size  # size of validation batch size
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    
    #### READ DATA ##################################################################    
    
    ### PREDICT ONLY SEA ICE U & V
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    ##########################################################################################
    train_list = torch.load(f'../Grid_graph_train_data.pt')
    val_list = torch.load(f'../Grid_graph_val_data.pt')   
    
    train_sampler, train_loader = make_sampler_and_loader(args, train_list) 
    val_sampler, val_loader = make_sampler_and_loader(args, val_list)

    print("######## TRAINING/VALIDATION DATA IS PREPARED ########")
    
    torch.cuda.empty_cache() 

    if device == "cpu":
        device_name = "cpu"
    else:
        device_name = "gpu"
    
    if args.model_type == "gcn":
        net = GCNet(4, 5)  # Graph convolutional network
    elif args.model_type == "egcn":
        net = EGCNet(4, 5)  # Equivariant Graph convolutional network
    
    model_name = f"torch_gcn_lr{lr}_{phy}_{device_name}"

    net.to(device)

    if args.no_cuda == False:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
        )

    if phy == "phy":
        loss_fn = physics_loss() # nn.L1Loss() #nn.CrossEntropyLoss()
    elif phy == "nophy":
        loss_fn = nn.MSELoss() #custom_loss() # nn.L1Loss() #nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=1.0)

    history = {'loss': [], 'val_loss': [], 'time': []}

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    
    t0 = time.time()

    ## Train model #############################################################
#     for n in range(0, n_epochs):
#         net.train()    
#         train_loss = 0.0
#         val_loss = 0.0

#         for data in loader:
#             optimizer.zero_grad()  # Clear gradients.
#             y_pred = net(torch.tensor(data.x, dtype=torch.float32).to(device), data.edge_index.to(device))  # Perform a single forward pass.
#             y_true = torch.tensor(data.y, dtype=torch.float32)
#             loss = loss_fn(y_pred.to(device), y_true.to(device))  # Compute the loss solely based on the training nodes.
#             loss.backward()  # Derive gradients.
#             optimizer.step()  # Update parameters based on gradients. 
#             train_loss += loss.item()

#         net.eval()

#         for val_data in val_loader:
#             y_pred = net(torch.tensor(val_data.x, dtype=torch.float32).to(device), val_data.edge_index.to(device))  # Perform a single forward pass.
#             y_true = torch.tensor(val_data.y, dtype = torch.float32)
#             val_loss += loss_fn(y_pred.to(device), y_true.to(device)).item()  # Compute the loss solely based on the training nodes.

#         history['loss'].append(train_loss/len(loader))
#         history['val_loss'].append(val_loss/len(val_loader))
#         history['time'].append(time.time()-t0)

#         if n % 2== 0:
#             print(n, "Epoch {0} - train: {1:.3f}, val: {2:.3f}".format(str(n).zfill(3), train_loss, val_loss))

#     torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

#     with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
#         pickle.dump(history, file)
    
    ## Train model (distributed parallel) ######################################
    for epoch in range(n_epochs):

        train_cnt = 0
        
        print(train_loader)
        
        train_loss = train(
            epoch,
            net,
            optimizer,
            loss_fn,
            train_loader,
            train_sampler,
            args
        )
        
        scheduler.step()
        val_loss = validate(epoch, net, loss_fn, val_loader, args)
        
        if dist.get_rank() == 0:
            if epoch % args.checkpoint_freq == 0:
                save_checkpoint(net.module, optimizer, args.checkpoint_format.format(epoch=epoch))
        
            history['loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['time'].append(time.time() - t0)
            
            if epoch == n_epochs-1:
                torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

                with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
                    pickle.dump(history, file)
    
    torch.cuda.empty_cache()
    
    # Test the model with the trained model ======================================== 
    
    if dist.get_rank() == 0:
        net.eval()

        scaling = [1, 5000, 5000, 5000, 4000]
        y_pred = np.zeros([len(val_list), 1112, 5])
        y_true = np.zeros([len(val_list), 1112, 5])
        count = 0

        rates = np.zeros(len(val_list))
        years = np.zeros(len(val_list))

        print("Train")
        for k in range(0, len(val_list)):
            data = val_list[k]
            r = data.x[0, 2]
            year = data.x[0, 3]*20
            if r not in rates:
                rates.append(r)
                print(r, year)
            prd = net(data.x.to(torch.float).to(device), data.edge_index.to(device)).to('cpu').detach().numpy()
            tru = data.y.to('cpu').detach().numpy()
            for i in range(0, prd.shape[1]):
                prd[:, i] = prd[:, i]*scaling[i]
                tru[:, i] = tru[:, i]*scaling[i]
            y_pred[k] = prd
            y_true[k] = tru

            rates[k] = r
            years[k] = year

            count += 1
            
        test_save = [rates, years, y_true.to('cpu').detach().numpy(), y_pred.to('cpu').detach().numpy()]

        with open(f'../results/test_{model_name}.pkl', 'wb') as file:
            pickle.dump(test_save, file)
            
    
                        
    if dist.get_rank() == 0:
        print("#### Validation done!! ####")     
    # ===============================================================================

if __name__ == '__main__':
    main()
