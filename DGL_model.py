import numpy as np

import torch    
import torch.nn as nn
import torch.nn.functional as F

# from dgl.nn.pytorch import GINConv, SumPooling    
from dgl.nn import GraphConv


## Graph Isomorphism Network (GIN) ========================
# class GIN(nn.Module):
#     def __init__(self, ch_input, ch_output, hidden_channels = 128):
#         super(GIN, self).__init__()
        
#         self.activation = nn.ReLU()
#         self.conv1 = GINConv(nn.Linear(ch_input, hidden_channels), aggregator_type='sum')
#         self.conv2 = GINConv(nn.Linear(hidden_channels, ch_output), aggregator_type='sum')
#         self.pool = SumPooling()

#     def forward(self, g, feats):
#         feats = self.conv1(g, feats)
#         feats = self.activation(feats)
#         feats = self.conv2(g, feats)

#         return self.pool(g, feats)

## Multi-layer perceptron ===============================
class MLP(nn.Module):
    
    def __init__(self, ch_input, ch_output, hidden_channels = 128):
        super(MLP, self).__init__()
        # torch.manual_seed(1234567)
        self.activation = nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        
        self.lin1 = torch.nn.Linear(ch_input, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, ch_output)

    def forward(self, g, in_feat):
        
        x = self.activation(self.lin1(in_feat));
        x = self.activation(self.lin2(x));
        x = self.lin3(x);
        
        return x
    
## Graph convolutional network =============================
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    
class FCNet(torch.nn.Module):
    def __init__(self, ch_input, ch_output, hidden_channels = 128):
        super().__init__()
        # torch.manual_seed(1234567)
        self.activation = nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        
        self.lin1 = torch.nn.Linear(ch_input, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin5 = torch.nn.Linear(hidden_channels, ch_output)
        
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, pos, edge_index):
        
        x = self.dropout(self.activation(self.lin1(x)));
        x = self.dropout(self.activation(self.lin2(x)));
        x = self.dropout(self.activation(self.lin3(x)));
        x = self.activation(self.lin4(x));
        x = self.lin5(x);

        
        return x
    

def index_sum(agg_size, source, idx, cuda):
    """
        source is N x hid_dim [float]
        idx    is N           [int]
        
        Sums the rows source[.] with the same idx[.];
    """
    tmp = torch.zeros((agg_size, source.shape[1]))
    tmp = tmp.cuda() if cuda else tmp
    res = torch.index_add(tmp, 0, idx, source)
    return res

class EGCNet(torch.nn.Module):
    def __init__(self, ch_input, ch_output, hidden_channels = 128, cuda=True):
        super().__init__()
        # torch.manual_seed(1234567)
        self.activation = nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        # self.emb = nn.Linear(ch_input, hidden_channels) 
        self.gnn = ConvEGNN(ch_input, hidden_channels, cuda=cuda)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin5 = torch.nn.Linear(hidden_channels, ch_output)
        
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, pos, edge_index):
        # x = self.emb(x)
        x = self.gnn(x, pos, edge_index); #self.conv1(x)
        x = self.dropout(self.activation(self.lin1(x)));
        x = self.dropout(self.activation(self.lin2(x)));
        x = self.dropout(self.activation(self.lin3(x)));
        x = self.dropout(self.activation(self.lin4(x)));
        x = self.lin5(x);
        
        return x
    
class ConvEGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.Tanh(),
            nn.Linear(hid_dim, hid_dim), nn.Tanh())
        
        # preducts "soft" edges based on messages 
        self.f_inf = nn.Sequential( 
            nn.Linear(hid_dim, 1),
            nn.Tanh())
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.Tanh(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, b, pos, edge_index):
        e_st = edge_index[0, :]
        e_end = edge_index[1, :]

        dists = torch.norm(torch.square(pos[e_st] - pos[e_end]), dim=1).reshape(-1, 1)
        
        # compute messages
        tmp = torch.hstack([b[e_st], b[e_end], dists])
        
        m_ij = self.f_e(tmp)
        
        # predict edges
        e_ij = self.f_inf(m_ij)
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = index_sum(b.shape[0], e_ij*m_ij, edge_index[0,:], self.cuda)
        
        # update hidden representations
        b += self.f_h(torch.hstack([b, m_i]))
        del tmp, m_ij, e_ij, m_i
        return b