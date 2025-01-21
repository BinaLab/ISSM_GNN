# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from datetime import datetime

import scipy.io as sio
from scipy.interpolate import griddata
import pickle

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import torch
from tqdm import tqdm
import glob, os
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

def norm(x, mean, std):
    return (x-mean)/std

## Dataset for train ===================================
class GNN_Helheim_Dataset(DGLDataset):
    def __init__(self, filename, initial = False):
        self.initial = initial
        super().__init__(name="pig", url = filename)
        
    def process(self):
        self.graphs = []
        files = self.url
        
        # # Region filtering
        # filename = f'D:\\ISSM\\Helheim\\Helheim_r100_030.mat'
        # test = sio.loadmat(filename)
        # mask = test['S'][0][0][11][0]

        first = True
        # "READING GRAPH DATA..."
        for filename in tqdm(files[:]):
            rate = int(filename[-11:-8])
            test = sio.loadmat(filename)

            xc = test['S'][0][0][0]
            yc = test['S'][0][0][1]
            elements = test['S'][0][0][2]-1
            idx = np.where((xc[:, 0]>-230000) & (yc[:, 0] < 2500000))[0] # Spatial filtering
            # idx = np.where((xc[:, 0]>230000) & (yc[:, 0] < -2500000))[0] # Spatial filtering
            xc = xc[idx]
            yc = yc[idx]

            mask = test['S'][0][0][11][:, idx]
            # ice = np.zeros(mask.shape) # Negative: ice; Positive: no-ice
            # ice[mask > 0] = 0.5 # ice = 0; no-ice = 1
            ice = np.where(mask < 0, mask / 1000000, mask/10000)

            # ice = np.where(mask < 0, mask / 200000, mask/4000)
            # ice[ice > 1] = 1.
            # ice[ice < -1] = -1.
            ice_mask = np.where(mask < 0, 1, 0)

            smb = test['S'][0][0][3][:, idx] # * ice_mask
            vx = test['S'][0][0][4][:, idx] # * ice_mask
            vy = test['S'][0][0][5][:, idx] # * ice_mask
            vel = test['S'][0][0][6][:, idx] # * ice_mask
            surface = test['S'][0][0][7][:, idx] # * ice_mask
            base = test['S'][0][0][8][:, idx]
            H = test['S'][0][0][9][:, idx] # * ice_mask
            f = test['S'][0][0][10][:, idx] # * ice_mask   

            if self.initial == "flow":
                sigmaVM = test['S'][0][0][14][:, idx]
                cr = test['S'][0][0][15][:, idx]
                mr = test['S'][0][0][18][:, idx]
                fc = test['S'][0][0][16][:, idx]

            n_year, n_sample = H.shape

            if first:

                src = []
                dst = []
                weight = []
                slope = []

                for i, i0 in enumerate(idx): #range(0, n_sample):        
                    p1, p2 = np.where(elements == i0)
                    connect = []
                
                    for p in p1:
                        for k0 in elements[p]:
                            if (k0 not in connect) and (k0 in idx) and (k0 != i0): 
                                k = np.where(idx == k0)[0][0]
                                connect.append(k0)
                                dist = ((xc[i]-xc[k])**2+(yc[i]-yc[k])**2)**0.5                                
                                weight.append(np.exp(-(dist/100)))
                                slope.append([np.exp(-(dist/1000)), 1-(base[0,i]-base[0,k])/dist, 1-(surface[0,i]-surface[0,k])/dist])
                                             # 1-(vx[0,i]-vx[0,k])/dist, 1-(vy[0,i]-vy[0,k])/dist])
                                src.append(int(i))
                                dst.append(int(np.where(idx == k0)[0][0]))

                # for i in range(0, n_sample):        
                #     p1, p2 = np.where(elements == i)
                #     connect = []

                #     for p in p1:
                #         for k in elements[p]:
                #             if (k != i) and (k not in connect):
                #                 connect.append(k)
                #                 dist = ((xc[i]-xc[k])**2+(yc[i]-yc[k])**2)**0.5                                
                #                 weight.append(np.exp(-(dist/1000)))
                #                 slope.append([np.exp(-(dist/1000)), (base[0,i]-base[0,k])/dist, (surface[0,i]-surface[0,k])/dist,
                #                              (vx[0,i]-vx[0,k])/dist, (vy[0,i]-vy[0,k])/dist]) 
                #                 src.append(int(i))
                #                 dst.append(int(k))

                src = torch.tensor(src)
                dst = torch.tensor(dst)
                weight = torch.tensor(weight)
                slope = torch.arctan(torch.tensor(slope))
                first = False
            else:
                pass                    

            for t in range(1, n_year):
                # INPUT: x/y coordinates, melting rate, time, SMB, Vx0, Vy0, Surface0, Base0, Thickness0, Floating0
                inputs = torch.zeros([n_sample, 12])
                # OUTPUT: Vx, Vy, Vel, Surface, Thickness, Floating
                outputs = torch.zeros([n_sample, 6])

                ## INPUTS (initial) ================================================
                if self.initial == "flow":
                    inputs[:, 0] = torch.tensor((xc[:, 0]-xc.min())/10000)
                    inputs[:, 1] = torch.tensor((yc[:, 0]-yc.min())/10000)
                    inputs[:, 2] = torch.tensor(smb[t-1, :]/20)
                    inputs[:, 3] = torch.tensor(surface[t-1, :]/5000) # Surface elevation
                    inputs[:, 4] = torch.tensor(base[t-1, :]/5000) # Base elevation
                    inputs[:, 5] = torch.tensor(fc[t-1, :]/12000) # Basal friction coefficient
                    inputs[:, 6] = torch.tensor(mr[t-1, :]/3000) # Ocean melting rate
                    inputs[:, 7] = torch.tensor(ice[t-1, :]) # Ice mask
                
                elif self.initial == "initial":
                    inputs[:, 0] = torch.tensor((xc[:, 0]-xc.min())/10000) # torch.tensor(xc[0, :]/10000) # torch.tensor((xc[:, 0]-xc.min())/(xc.max()-xc.min())) # X coordinate
                    inputs[:, 1] = torch.tensor((yc[:, 0]-yc.min())/10000) # torch.tensor(yc[0, :]/10000) # torch.tensor((yc[:, 0]-yc.min())/(yc.max()-yc.min())) # Y coordinate
                    inputs[:, 2] = torch.tensor((rate-50)/(150-50)) # Sigma_max
                    inputs[:, 3] = torch.tensor(t/n_year) # Year
                    inputs[:, 4] = torch.tensor(smb[t, :]/20) # Surface mass balance
                    inputs[:, 5] = torch.tensor(vx[0, :]/10000) # Initial Vx
                    inputs[:, 6] = torch.tensor(vy[0, :]/10000) # Initial Vy
                    inputs[:, 7] = torch.tensor(vel[0, :]/10000) # Initial Velocity
                    inputs[:, 8] = torch.tensor(surface[0, :]/5000) # Initial surface elevation
                    inputs[:, 9] = torch.tensor(base[0, :]/5000) # Initial base elevation
                    inputs[:, 10] = torch.tensor(H[0, :]/5000) # Initial ice thickness
                    # inputs[:, 11] = torch.tensor(f[0, :]/5000) # Initial floating part
                    inputs[:, 11] = torch.tensor(ice[0, :]) # Initial ice mask

                ## INPUTS (previous timestep) ================================================
                else:
                    inputs[:, 0] = torch.tensor((xc[:, 0]-xc.min())/10000) # torch.tensor(xc[0, :]/10000) # torch.tensor((xc[:, 0]-xc.min())/(xc.max()-xc.min())) # X coordinate
                    inputs[:, 1] = torch.tensor((yc[:, 0]-yc.min())/10000) # torch.tensor(yc[0, :]/10000) # torch.tensor((yc[:, 0]-yc.min())/(yc.max()-yc.min())) # Y coordinate
                    inputs[:, 2] = torch.tensor((rate-50)/(150-50) * ice_mask[t-1, :])  # Sigma_max
                    inputs[:, 3] = torch.tensor(t/n_year) # Year
                    inputs[:, 4] = torch.tensor(smb[t-1, :]/20) # Surface mass balance
                    inputs[:, 5] = torch.tensor(vx[t-1, :]/10000) # t-1 Vx
                    inputs[:, 6] = torch.tensor(vy[t-1, :]/10000) # t-1 Vy
                    inputs[:, 7] = torch.tensor(vel[t-1, :]/10000) # t-1 Velocity
                    inputs[:, 8] = torch.tensor(surface[0, :]/5000) # Initial surface elevation
                    inputs[:, 9] = torch.tensor(base[0, :]/5000) # Initial base elevation
                    inputs[:, 10] = torch.tensor(H[0, :]/5000) # Initial ice thickness
                    # inputs[:, 11] = torch.tensor(f[0, :]/5000) # Initial floating part
                    inputs[:, 11] = torch.tensor(ice[t-1, :]) # t-1 ice mask
                    inputs[:, 12] = torch.tensor(ice[t-1, :]) # t-1 ice mask
                
                ###### Normalization #####################                
                # vx_mean = torch.mean(inputs[:, 5])
                # vy_mean = torch.mean(inputs[:, 6])
                # vel_mean = torch.mean(inputs[:, 7])
                # surf_mean = torch.mean(inputs[:, 8])
                # base_mean = torch.mean(inputs[:, 9])
                # H_mean = torch.mean(inputs[:, 10])

                ## OUTPUTS ===============================================
                if self.initial == "flow":
                    outputs[:, 0] = torch.tensor(vx[t, :]/10000) # Initial Vx
                    outputs[:, 1] =  torch.tensor(vy[t, :]/10000) # Initial Vx
                    outputs[:, 2] = torch.tensor(sigmaVM[t, :]/(1.5*1e6)) # Initial surface elevation
                    
                else:
                    outputs[:, 0] = torch.tensor(vx[t, :]/10000) # Initial Vx
                    outputs[:, 1] =  torch.tensor(vy[t, :]/10000) # Initial Vx
                    outputs[:, 2] = torch.tensor(vel[t, :]/10000) # Initial surface elevation
                    outputs[:, 3] = torch.tensor(surface[t, :]/5000) # Initial surface elevation
                    outputs[:, 4] = torch.tensor(H[t, :]/5000) # Initial ice thickness
                    # outputs[:, 5] = torch.tensor(f[t, :]/5000) # Initial floating part 
                    outputs[:, 5] = torch.tensor(ice[t, :]) # Initial floating part 

                # for i in range(0, n_sample):        
                #     inputs[i, :] = torch.tensor([(xc[i, 0]-xc.min())/(xc.max()-xc.min()), (yc[i, 0]-yc.min())/(yc.max()-yc.min()), rate*0.001, t/n_year, smb[t,i],
                #                                  vx[0, i]/5000, vy[0, i]/5000, surface[0, i]/4000, base[0,i]/4000, H[0,i]/4000, f[0,i]/3000
                #                                 ])
                #     outputs[i, :] = torch.tensor([vx[t, i]/5000, vy[t, i]/5000, vel[t,i]/5000, surface[t, i]/4000, H[t,i]/4000, f[t,i]/3000])

                g = dgl.graph((src, dst), num_nodes=n_sample)
                g.ndata['feat'] = inputs
                g.ndata['label'] = outputs
                g.edata['weight'] = weight
                g.edata['slope'] = slope

                self.graphs.append(g)
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
def generate_list(region = "Helheim", folder = "../data", model = "gnn"):
    ## MAKE TRAINING AND TESTING DATASETS FOR GNN
    train_files = []
    val_files = []
    test_files = []
    
    if region == "Helheim":
        if model == "gnn":
            filelist = glob.glob(f'{folder}/Helheim_r*_030.mat')
        elif model == "cnn":
            filelist = glob.glob(f'{folder}/Helheim_r*_030_CNN_200m.pkl')
        for f in sorted(filelist):
            rate = f.split("_r")[1][:3]
            if (int(rate) >= 70) and (int(rate) <= 100) and (rate != "080"):
                # train_files.append(f)
                if rate == "075" or rate == "095": #int(f[-11:-8])%10 == 5: # f[-11:-8] == "070" or f[-11:-8] == "080" or f[-11:-8] == "115" or f[-11:-8] == "115":
                    val_files.append(f)
                    test_files.append(f)
                # elif f[-11:-8] == "085" or f[-11:-8] == "105" or f[-11:-8] == "125":
                #     test_files.append(f)
                else:
                    train_files.append(f)
                    
    elif region == "PIG":
        if model == "gnn":
            filelist = glob.glob(f'{folder}/PIG_transient_m*_r*.mat')
        elif model == "cnn":
            filelist = glob.glob(f'{folder}/PIG_transient_m*_r*_CNN.pkl')
        for f in sorted(filelist):
            rate = int(f.split("_r")[1][:3])
            if rate % 20 == 0:
                test_files.append(f)
            elif rate % 20 == 10:
                val_files.append(f)
            else:
                train_files.append(f)
    
    return train_files, val_files, test_files

## Dataset for train ===================================
class GNN_PIG_Dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        self.graphs = []
        files = self.url
        
        # # Region filtering
        # filename = f'D:\\ISSM\\Helheim\\Helheim_r100_030.mat'
        # test = sio.loadmat(filename)
        # mask = test['S'][0][0][11][0]

        first = True
        # "READING GRAPH DATA..."
        for filename in tqdm(files[:]):
            mesh = int(filename.split("_m")[1][:3])
            rate = int(filename.split("_r")[1][:3])
            test = sio.loadmat(filename)

            xc = test['S'][0][0][0]
            yc = test['S'][0][0][1]
            elements = test['S'][0][0][2]-1
            smb = test['S'][0][0][3]
            vx = test['S'][0][0][4]
            vy = test['S'][0][0][5]
            vel = test['S'][0][0][6]
            surface = test['S'][0][0][7]
            base = test['S'][0][0][8]
            H = test['S'][0][0][9]
            f = test['S'][0][0][10]
            # mask = test['S'][0][0][11]
            # ice = np.zeros(mask.shape) # Negative: ice; Positive: no-ice
            # ice[mask > 0] = 0.5 # ice = 0; no-ice = 1
            # ice = np.where(mask < 0, mask / 1000000, mask/10000)

            n_year, n_sample = H.shape
            
            if first:
                mesh0 = mesh
            elif mesh0 != mesh:
                first = True
                mesh0 = mesh

            if first:
                src = []
                dst = []
                weight = []
                slope = []

                for i in range(0, n_sample):        
                    p1, p2 = np.where(elements == i)
                    connect = []

                    for p in p1:
                        for k in elements[p]:
                            if (k != i) and (k not in connect):
                                connect.append(k)
                                dist = ((xc[i]-xc[k])**2+(yc[i]-yc[k])**2)**0.5                                
                                weight.append(np.exp(-(dist/1000)))
                                slope.append([np.exp(-(dist/1000)), (base[0,i]-base[0,k])/dist, (surface[0,i]-surface[0,k])/dist,
                                             (vx[0,i]-vx[0,k])/dist, (vy[0,i]-vy[0,k])/dist]) 
                                src.append(int(i))
                                dst.append(int(k))

                src = torch.tensor(src)
                dst = torch.tensor(dst)
                weight = torch.tensor(weight)
                slope = torch.arctan(torch.tensor(slope))
                first = False
            else:
                pass                    

            for t in range(0, n_year):
                # INPUT: x/y coordinates, melting rate, time, SMB, Vx0, Vy0, Surface0, Base0, Thickness0, Floating0
                inputs = torch.zeros([n_sample, 12])
                # OUTPUT: Vx, Vy, Vel, Surface, Thickness, Floating
                outputs = torch.zeros([n_sample, 6])

                ## INPUTS ================================================
                inputs[:, 0] = torch.tensor((xc[:, 0]-xc.min())/10000) # torch.tensor(xc[0, :]/10000) # torch.tensor((xc[:, 0]-xc.min())/(xc.max()-xc.min())) # X coordinate
                inputs[:, 1] = torch.tensor((yc[:, 0]-yc.min())/10000) # torch.tensor(yc[0, :]/10000) # torch.tensor((yc[:, 0]-yc.min())/(yc.max()-yc.min())) # Y coordinate
                inputs[:, 2] = torch.where(torch.tensor(f[0, :]) < 0, rate/100, 0) # Melting rate (0-100)
                inputs[:, 3] = torch.tensor(t/n_year) # Year
                inputs[:, 4] = torch.tensor(smb[t, :]/20) # Surface mass balance
                inputs[:, 5] = torch.tensor(vx[0, :]/10000) # Initial Vx
                inputs[:, 6] = torch.tensor(vy[0, :]/10000) # Initial Vx
                inputs[:, 7] = torch.tensor(vel[0, :]/10000) # Initial Vel
                inputs[:, 8] = torch.tensor(surface[0, :]/5000) # Initial surface elevation
                inputs[:, 9] = torch.tensor(base[0, :]/5000) # Initial base elevation
                inputs[:, 10] = torch.tensor(H[0, :]/5000) # Initial ice thickness
                inputs[:, 11] = torch.tensor(f[0, :]/5000) # Initial floating part
                # inputs[:, 11] = torch.tensor(ice[0, :]) # Initial ice mask

                ## OUTPUTS ===============================================
                outputs[:, 0] = torch.tensor(vx[t, :]/10000) # Initial Vx
                outputs[:, 1] =  torch.tensor(vy[t, :]/10000) # Initial Vx
                outputs[:, 2] = torch.tensor(vel[t, :]/10000) # Initial surface elevation
                outputs[:, 3] = torch.tensor(surface[t, :]/5000) # Initial base elevation
                outputs[:, 4] = torch.tensor(H[t, :]/5000) # Initial ice thickness
                outputs[:, 5] = torch.tensor(f[t, :]/5000) # Initial floating part 
                # outputs[:, 5] = torch.tensor(ice[t, :]) # Initial floating part 

                # for i in range(0, n_sample):        
                #     inputs[i, :] = torch.tensor([(xc[i, 0]-xc.min())/(xc.max()-xc.min()), (yc[i, 0]-yc.min())/(yc.max()-yc.min()), rate*0.001, t/n_year, smb[t,i],
                #                                  vx[0, i]/5000, vy[0, i]/5000, surface[0, i]/4000, base[0,i]/4000, H[0,i]/4000, f[0,i]/3000
                #                                 ])
                #     outputs[i, :] = torch.tensor([vx[t, i]/5000, vy[t, i]/5000, vel[t,i]/5000, surface[t, i]/4000, H[t,i]/4000, f[t,i]/3000])

                g = dgl.graph((src, dst), num_nodes=n_sample)
                g.ndata['feat'] = inputs
                g.ndata['label'] = outputs
                g.edata['weight'] = weight
                g.edata['slope'] = slope

                self.graphs.append(g)
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)

class CNN_PIG_Dataset(Dataset):
    def __init__(self, files):
        
        self.input = torch.tensor([])
        self.output = torch.tensor([])

        first = True
        # "READING GRAPH DATA..."
        for filename in tqdm(files[:]):
            
            with open(filename, 'rb') as file:
                [input0, output0] = pickle.load(file)
            
            rate = int(filename.split("_r")[1][:3])
            input0 = torch.tensor(input0, dtype=torch.float32)
            output0 = torch.tensor(output0, dtype=torch.float32)
            
            if first:
                self.input = input0
                self.output = output0
                first = False
            else:
                self.input = torch.cat((self.input, input0), dim = 0)
                self.output = torch.cat((self.output, output0), dim = 0)
        
    def __getitem__(self, i):
        cnn_input = self.input[i]
        cnn_input[torch.isnan(cnn_input)] = 0        
        cnn_output = self.output[i]
        cnn_output[torch.isnan(cnn_output)] = 0
        return (cnn_input, cnn_output)
    
    def __len__(self):
        return len(self.output)

class CNN_Helheim_Dataset(Dataset):
    def __init__(self, files):
        
        self.input = torch.tensor([])
        self.output = torch.tensor([])

        first = True
        # "READING GRAPH DATA..."
        for filename in tqdm(files[:]):
            
            with open(filename, 'rb') as file:
                [input0, output0] = pickle.load(file)
            
            rate = int(filename.split("_r")[1][:3])
            input0 = torch.tensor(input0, dtype=torch.float32) #torch.tensor(input0[:, :, 1:108, 129:214], dtype=torch.float32)
            output0 = torch.tensor(output0, dtype=torch.float32) # torch.tensor(output0[:, :, 1:108, 129:214], dtype=torch.float32)
            
            if first:
                self.input = input0
                self.output = output0
                first = False
            else:
                self.input = torch.cat((self.input, input0), dim = 0)
                self.output = torch.cat((self.output, output0), dim = 0)
        
    def __getitem__(self, i):
        cnn_input = self.input[i]
        cnn_input[torch.isnan(cnn_input)] = 0        
        cnn_output = self.output[i]
        cnn_output[torch.isnan(cnn_output)] = 0
        return (cnn_input, cnn_output)
    
    def __len__(self):
        return len(self.output)
    
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

def MAE(prd, obs):
    return np.nanmean(abs(obs-prd))

def MAE_grid(prd, obs):
    err = abs(obs-prd)
    return np.nanmean(err, axis=0)

def RMSE(prd, obs):
    err = np.square(obs-prd)
    return np.nanmean(err)**0.5

def RMSE_grid(prd, obs):
    err = np.square(obs-prd)
    return np.nanmean(err, axis=0)**0.5

def corr_grid(prd, obs):
    r1 = np.nansum((prd-np.nanmean(prd))*(obs-np.nanmean(obs)),axis=0)
    r2 = np.nansum(np.square(prd-np.nanmean(prd)), axis=0)*np.nansum(np.square(obs-np.nanmean(obs)),axis=0)
    r = r1/r2**0.5
    return r

def skill(prd, obs):
    err = np.nanmean(np.square(prd-obs))**0.5/np.nanmean(np.square(obs-np.nanmean(obs)))**0.5
    return 1-err

def MBE(prd, obs):
    return np.nanmean(prd-obs)

def corr(prd, obs):
    prd = prd.flatten()
    obs = obs.flatten()
    
    r = np.ma.corrcoef(np.ma.masked_invalid(prd), np.ma.masked_invalid(obs))[0, 1]
    return r

def sort_xy(x, y):
    
    print(len(x))

    x0 = x[0] #200000 #np.median(x)
    y0 = y[0] #-2450000 #np.median(y)
    
    x_sorted = []
    y_sorted = []
    
    i = 0

    while len(x_sorted) < len(x):      
        
        dist = ((x-x[i])**2 + (y-y[i])**2)**0.5
        cand = np.argsort(dist)       
        
        r = np.sqrt((x[cand]-x0)**2 + (y[cand]-y0)**2)
        angles = np.where((y[cand]-y0) > 0, np.arccos((x[cand]-x0)/r), 2*np.pi-np.arccos((x[cand]-x0)/r))     
        
        k1 = cand[0] #np.argsort(angles)[0]
        k2 = cand[1] #np.argsort(angles)[1]
        
        for c in cand:
            if x[c] not in x_sorted:
                x_sorted.append(x[c])
                y_sorted.append(y[c])
                i = c
                break
                
    return x_sorted, y_sorted

def triangle_area(x1, x2, x3, y1, y2, y3):
    A = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) * 0.5
    return A

def area_mean(xc, yc, elements, value):
    area_sum = 0
    area = 0
    for i in range(0, elements.shape[0]):
        p1 = elements[i, 0]
        p2 = elements[i, 1]
        p3 = elements[i, 2]
        A = triangle_area(xc[p1], xc[p2], xc[p3], yc[p1], yc[p2], yc[p3])[0]
        M = (value[p1] + value[p2] + value[p3])/3
        area_sum += M*A
        area += A
    area_mean = area_sum / area
    return area_sum, area_mean

def node_area(xc, yc, elements):
    n_area = np.zeros(xc.shape)
    for i in range(0, xc.shape[0]):        
        p1, p2 = np.where(elements == i)

        for p in p1:
            pi = elements[p, :]
            n_area[i] += triangle_area(xc[pi[0], 0], xc[pi[1], 0], xc[pi[2], 0], yc[pi[0], 0], yc[pi[1], 0], yc[pi[2], 0])
        n_area[i] = n_area[i] / len(p1)
    return n_area

def approx_area(xc, yc, elements):
    area = np.zeros(xc.shape[0])
    for i in range(0, xc.shape[0]):
        A = 0
        p1, p2 = np.where(elements == i)
        for p in p1:
            A += triangle_area(xc[elements[p,0]], xc[elements[p,1]], xc[elements[p,2]], yc[elements[p,0]], yc[elements[p,1]], yc[elements[p,2]])
        A = A/len(p1)
        area[i] = A
    return area / 1e6

def add_vel(y_true0):
    vel = np.expand_dims((y_true0[:, :, 0]**2 + y_true0[:, :, 1]**2)**0.5, axis = 2)
    y_true0 = np.append(y_true0, vel, axis = 2)
    return y_true0

def convert_binary(levelset, thickness, threshold = 0):
    levelset = torch.tensor(levelset)
    output = torch.zeros(levelset.shape)
    output = torch.where(((levelset > threshold) | (thickness < 20/5000)), 0., 1.) # positive: water, negative: ice
    return output