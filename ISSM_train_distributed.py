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

# try:
#     from torch.cuda.amp import GradScaler

#     TORCH_FP16 = True
# except ImportError:
#     TORCH_FP16 = False


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
        '--data-dir',
        type=str,
        default='../data/', #'D:\\PINN\\data\\',
        metavar='D',
        help='directory to download dataset to',
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='train_cnn_2018_2022_v5.pkl',
        help='filename of dataset',
    )    
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
        '--date',
        type=int,
        default=2022,
        help='year to exclude during the training process',
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
        default='phy',
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
        default=0.001,
        metavar='LR',
        help='base learning rate (default: 0.01)',
    )
    parser.add_argument(
        '--day-int',
        type=int,
        default=1,
        help='date interval to create inputs',
    )
    parser.add_argument(
        '--forecast',
        type=int,
        default=1,
        help='date to forecast',
    )
    parser.add_argument(
        '--predict',
        type=str,
        default="all",
        help='prediction outputs',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="unet",
        help='types of the neural network model (e.g. unet, cnn, fc)',
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )

    # Set automatically by torch distributed launch
    # parser.add_argument(
    #     '--local_rank',
    #     type=int,
    #     default=0,
    #     help='local rank for distributed training',
    # )
    
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
    # val_sampler = DistributedSampler(
    #     val_dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=dist.get_rank(),
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.val_batch_size,
    #     sampler=val_sampler,
    #     **kwargs,
    # )

    return train_sampler, train_loader

# def init_processes(backend):
#     dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
#     run(backend)
    
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
        for batch_idx, (data, target) in enumerate(train_loader):
            mini_step += 1
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            if args.phy == "phy":
                loss = loss_func(output, target, data[:, 2, :, :].cuda())
            else:
                loss = loss_func(output, target)

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
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                
                if args.phy == "phy":
                    val_loss.update(loss_func(output, target, data[:, 2, :, :].cuda()))
                else:
                    val_loss.update(loss_func(output, target))

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
    
    data_path = args.data_dir
    data_file = args.data_file
    model_dir = args.model_dir
    date = args.date   

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    val_batch = args.val_batch_size  # size of validation batch size
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    dayint = args.day_int
    forecast = args.forecast    
    
    #### READ DATA ##################################################################    
    data_ver = data_file[-6:-4]
    
    with open(data_path + data_file, 'rb') as file:
        xx, yy, days, months, years, cnn_input, cnn_output = pickle.load(file)   
    
    if data_ver == 'v5':
        cnn_input = cnn_input[:,:,:,[0,1,2,4,5]]
        cnn_output = cnn_output[:,:,:,:-1]
    if data_ver == 'v6':
        cnn_input = cnn_input[:,:,:,[0,1,2,3,4,5]]
        cnn_output = cnn_output[:,:,:,:-1]
        
    if args.model_type == "mtunet":
        args.predict = "all"
        
    if args.predict == "sic":
        cnn_output = cnn_output[:,:,:,2:3]
    elif args.predict == "sit":
        if data_ver == 'v4':
            cnn_output = cnn_output[:,:,:,3:4]
        else:
            print(f"SIT prediction is not available with {data_ver} data >>> Proceed with all prediction")
    elif args.predict == "uv":
        cnn_output = cnn_output[:,:,:,0:2]     
        
    # Read landmask data
    with open(data_path + f"landmask_320.pkl", 'rb') as file:
        landmask = pickle.load(file) 
    landmask = torch.tensor(landmask) # Land = 1; Ocean = 0;
    if args.cuda:
        landmask = landmask.cuda() # Land = 1; Ocean = 0;
    
    # cnn_input = cnn_input[:, :, :, :4] # Only U, V, SIC, SIT as input
    cnn_input, cnn_output, days, months, years = convert_cnn_input2D(cnn_input, cnn_output, days, months, years, dayint, forecast)
    
    ## Add x y coordinates as inputs
    if args.model_type != "lg":
        xx_n = (xx - xx.min())/(xx.max() - xx.min()).astype(np.float16)
        yy_n = (yy - yy.min())/(yy.max() - yy.min()).astype(np.float16)
        cnn_input = np.concatenate((cnn_input, np.repeat(np.array([np.expand_dims(xx_n, 2)]), cnn_input.shape[0], axis = 0).astype(np.float16)), axis = 3)
        cnn_input = np.concatenate((cnn_input, np.repeat(np.array([np.expand_dims(yy_n, 2)]), cnn_input.shape[0], axis = 0).astype(np.float16)), axis = 3)
    
    ## Convert numpy array into torch tensor
    cnn_input = torch.tensor(cnn_input, dtype=torch.float32)
    cnn_output = torch.tensor(cnn_output, dtype=torch.float32)
    
    mask1 = (years == date) # Test samples
    mask2 = (days % 7 == 2) # Validation samples

    val_input = cnn_input[mask1] #cnn_input[(~mask1)&(mask2), :, :, :]
    val_output = cnn_output[mask1] #cnn_output[(~mask1)&(mask2), :, :, :]
    train_input = cnn_input[(~mask1)&(~mask2)] #cnn_input[(~mask1)&(~mask2), :, :, :]
    train_output = cnn_output[(~mask1)&(~mask2)] #cnn_output[(~mask1)&(~mask2), :, :, :]
    # test_input = cnn_input[mask1, :, :, :]
    # test_output = cnn_output[mask1, :, :, :]    
        
    if args.model_type == "fc": # in case of fully-connected layer: flatten layers
        train_input = torch.flatten(train_input[:, landmask==0, :], start_dim=0, end_dim= -2)
        train_output = torch.flatten(train_output[:, landmask==0, :], start_dim=0, end_dim= -2)
        val_input = torch.flatten(val_input, start_dim=0, end_dim= -2)
        val_output = torch.flatten(val_output, start_dim=0, end_dim= -2)
        n_samples, in_channels = train_input.size()
        _, out_channels = train_output.size()
        
    else:
        train_input = torch.permute(train_input, (0, 3, 1, 2))
        train_output = torch.permute(train_output, (0, 3, 1, 2))
        val_input = torch.permute(val_input, (0, 3, 1, 2))
        val_output = torch.permute(val_output, (0, 3, 1, 2))
        
        n_samples, in_channels, row, col = train_input.size()
        _, out_channels, _, _ = train_output.size()
    
    print(train_input.size(), train_output.size(), val_input.size(), val_output.size()) 
    
    train_dataset = TensorDataset(train_input, train_output)
    val_dataset = TensorDataset(val_input, val_output)
    # test_dataset = TensorDataset(test_input, test_output)
    
    train_sampler, train_loader = make_sampler_and_loader(args, train_dataset) 
    val_sampler, val_loader = make_sampler_and_loader(args, val_dataset)
    
    del cnn_input, cnn_output, train_input, train_output
    
    #############################################################################   
    if args.model_type == "unet":
        net = UNet(in_channels, out_channels)
    elif args.model_type == "mtunet":
        net = HF_UNet(in_channels, out_channels)
    elif args.model_type == "tsunet":
        net = TS_UNet(in_channels, out_channels) # Triple sharing
    elif args.model_type == "isunet":
        net = IS_UNet(in_channels, out_channels) # information sharing
    elif args.model_type == "lbunet":
        net = LB_UNet(in_channels, out_channels)
    elif args.model_type == "ebunet":
        net = EB_UNet(in_channels, out_channels)
    elif args.model_type == "cnn":
        net = Net(in_channels, out_channels)
    elif args.model_type == "fc":
        net = FC(in_channels, out_channels)
    elif args.model_type == "lg": # linear regression
        net = linear_regression(in_channels, out_channels, row, col)

    model_name = f"torch_{args.model_type}_{data_ver}_{args.predict}_wo{date}_{phy}_d{dayint}f{forecast}_{device_name}{world_size}"  

    # print(device)
    net.to(device)
    
    if args.no_cuda == False:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
        )

    if phy == "phy":
        loss_fn = physics_loss(landmask) # nn.L1Loss() #nn.CrossEntropyLoss()
    elif phy == "nophy":
        if args.model_type == "fc":
            loss_fn = nn.L1Loss()
        else:
            if args.predict== "all":
                loss_fn = custom_loss(landmask) # nn.L1Loss() #nn.CrossEntropyLoss()            
            else:
                loss_fn = single_loss(landmask)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=1.0)

    history = {'loss': [], 'val_loss': [], 'time': []}

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    
    t0 = time.time()
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        train_cnt = 0
        
        net.train()
        
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
    
    del train_dataset, train_loader, train_sampler, val_dataset, val_loader, val_sampler
    
    # Test the model with the trained model ========================================
    val_months = months[mask1]
    val_days = days[mask1]
    
    net.eval()
    
    for m in np.unique(val_months):
        if m % 3 == dist.get_rank():
            
            data = val_input[val_months==m, :, :, :]
            target = val_output[val_months==m, :, :, :]
            output = torch.zeros(target.size())
            
            with tqdm(total=target.size()[0],
                      bar_format='{l_bar}{bar:10}|{postfix}',
                      desc=f'Validation {date}-{str(int(m)).zfill(2)}'
                     ) as t:
                with torch.no_grad():
                    for j in range(0, target.size()[0]):
                        output[j, :, :, :] = net(data[j:j+1, :, :, :])
                        t.update(1)                  
                    
                    test_save = [data.to('cpu').detach().numpy(), target.to('cpu').detach().numpy(), output.to('cpu').detach().numpy(),
                                 val_months[val_months==m], val_days[val_months==m]]

                    # Open a file and use dump()
                    with open(f'../results/test_{model_name}_{str(int(m)).zfill(2)}.pkl', 'wb') as file:
                        pickle.dump(test_save, file)
                        
    if dist.get_rank() == 0:
        print("#### Validation done!! ####")     
    # ===============================================================================

if __name__ == '__main__':
    main()
