# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from datetime import datetime

import scipy.io as sio

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import torch
from tqdm import tqdm
import glob, os

## Dataset for train ===================================
class GNN_Helheim_Dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        self.graphs = []
        files = self.url
        
        first = True
        # "READING GRAPH DATA..."
        for filename in tqdm(files[:]):
            rate = int(filename[-11:-8])*10
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
            mask = test['S'][0][0][11]
            ice = np.ones(mask.shape)
            ice[mask > 0] = -1 # ice = 1; no-ice = -1

            n_year, n_sample = H.shape

            if first:

                src = []
                dst = []
                weight = []

                for i in range(0, n_sample):        
                    p1, p2 = np.where(elements == i)
                    connect = []

                    for p in p1:
                        for k in elements[p]:
                            if (k != i) and (k not in connect):
                                # connect.append(k)
                                dist = ((xc[i]-xc[k])**2+(yc[i]-yc[k])**2)**0.5
                                weight.append(np.exp(-(dist/1000)))
                                src.append(int(i))
                                dst.append(int(k))

                src = torch.tensor(src)
                dst = torch.tensor(dst)
                weight = torch.tensor(weight)
                first = False
            else:
                pass                    

            for t in range(0, n_year):
                # INPUT: x/y coordinates, melting rate, time, SMB, Vx0, Vy0, Surface0, Base0, Thickness0, Floating0
                inputs = torch.zeros([n_sample, 12])
                # OUTPUT: Vx, Vy, Vel, Surface, Thickness, Floating
                outputs = torch.zeros([n_sample, 6])

                ## INPUTS ================================================
                inputs[:, 0] = torch.tensor((xc[:, 0]-xc.min())/(xc.max()-xc.min())) # X coordinate
                inputs[:, 1] = torch.tensor((yc[:, 0]-yc.min())/(yc.max()-yc.min())) # Y coordinate
                inputs[:, 2] = torch.tensor(rate*0.001) # Melting rate
                inputs[:, 3] = torch.tensor(t/n_year) # Year
                inputs[:, 4] = torch.tensor(smb[t, :]) # Surface mass balance
                inputs[:, 5] = torch.tensor(vx[0, :]/5000) # Initial Vx
                inputs[:, 6] = torch.tensor(vy[0, :]/5000) # Initial Vx
                inputs[:, 7] = torch.tensor(vel[0, :]/5000) # Initial Vx
                inputs[:, 8] = torch.tensor(surface[0, :]/4000) # Initial surface elevation
                inputs[:, 9] = torch.tensor(base[0, :]/4000) # Initial base elevation
                inputs[:, 10] = torch.tensor(H[0, :]/4000) # Initial ice thickness
                # inputs[:, 11] = torch.tensor(f[0, :]/3000) # Initial floating part
                inputs[:, 11] = torch.tensor(ice[0, :]) # Initial ice mask

                ## OUTPUTS ===============================================
                outputs[:, 0] = torch.tensor(vx[t, :]/5000) # Initial Vx
                outputs[:, 1] =  torch.tensor(vy[t, :]/5000) # Initial Vx
                outputs[:, 2] = torch.tensor(vel[t, :]/5000) # Initial surface elevation
                outputs[:, 3] = torch.tensor(surface[t, :]/4000) # Initial base elevation
                outputs[:, 4] = torch.tensor(H[t, :]/4000) # Initial ice thickness
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

                self.graphs.append(g)
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
def generate_list():
    ## MAKE TRAINING AND TESTING DATASETS FOR GNN
    train_files = []
    val_files = []
    test_files = []

    for f in sorted(glob.glob(f'../data/Helheim_r*_030.mat')):
        if f[-11:-8] == "075" or f[-11:-8] == "095" or f[-11:-8] == "115":
            val_files.append(f)
        elif f[-11:-8] == "085" or f[-11:-8] == "105" or f[-11:-8] == "125":
            test_files.append(f)
        else:
            train_files.append(f)
    
    return train_files, val_files, test_files