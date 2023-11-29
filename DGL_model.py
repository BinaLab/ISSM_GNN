import numpy as np

import torch    
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GINConv, SumPooling    
from dgl.nn import DenseGraphConv, GraphConv, GATConv, SAGEConv, DenseSAGEConv, ChebConv, DenseChebConv, EGNNConv
from dgl import function as fn

class physics_loss(nn.Module):
    def __init__(self, landmask):
        super(physics_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd, sic0):
        
        sic_th = 0.001
        
        sic_p = prd[:, 2, :, :]
        # sic_p[sic_p > 1] = 1
        # sic_p[sic_p < 0] = 0
        sic_o = obs[:, 2, :, :]
        u_o = obs[:, 0, :, :]*50; v_o = obs[:, 1, :, :]*50
        u_p = prd[:, 0, :, :]*50; v_p = prd[:, 1, :, :]*50
        
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        # u_p[sic_p <= sic_th] = 0
        # v_p[sic_p <= sic_th] = 0
        
        err_u = torch.square(u_o - u_p) #[sic > 0]
        err_v = torch.square(v_o - v_p) #[sic > 0]
        
        sicmask = torch.max(sic_o, dim=0)[0]
        err1 = torch.mean(err_u + err_v, dim=0)[torch.where(self.landmask == 0)]
        err_sum = torch.mean(err1)

        err_sic = torch.square(sic_o - sic_p)
        
        err2 = torch.mean(err_sic, dim=0)[torch.where(self.landmask == 0)]
        err_sum += torch.mean(err2)*2500

class CNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_nodes, nrow, ncol, n_filters=128, kernel = 5):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_inputs, n_filters, kernel, padding = "same"),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )        
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel, padding = "same"),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel, padding = "same"),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel, padding = "same"),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel, padding = "same"),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )     
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=int(n_filters*(nrow//(2**5))*(ncol//(2**5))), out_features=n_outputs * n_nodes)


    def forward(self, x):
        
        x = self.activation(self.conv1(x))
        # print(x.shape)
        x = self.activation(self.conv2(x))
        # print(x.shape)
        x = self.activation(self.conv3(x))
        # print(x.shape)
        x = self.activation(self.conv4(x))
        # print(x.shape)
        x = self.activation(self.conv5(x))
        # print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.reshape(-1, self.n_nodes, self.n_outputs)
        
        return x
        
## Graph Isomorphism Network (GIN) ========================
class GIN(nn.Module):
    def __init__(self, ch_input, ch_output, hidden_channels = 128):
        super(GIN, self).__init__()
        
        self.activation = nn.LeakyReLU()
        self.conv1 = GINConv(nn.Linear(ch_input, hidden_channels), aggregator_type='sum')
        self.conv2 = GINConv(nn.Linear(hidden_channels, hidden_channels), aggregator_type='sum')
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin5 = torch.nn.Linear(hidden_channels, ch_output)
        # self.pool = SumPooling()

    def forward(self, g, feats):
        feats = self.activation(self.conv1(g, feats))
        feats = self.activation(self.conv2(g, feats))
        feats = self.activation(self.lin1(feats));
        feats = self.activation(self.lin2(feats));
        feats = self.activation(self.lin3(feats));
        feats = self.activation(self.lin4(feats));
        feats = self.lin5(feats);

        return feats

## Multi-layer perceptron ===============================
class MLP(nn.Module):
    
    def __init__(self, ch_input, ch_output, hidden_channels = 128):
        super(MLP, self).__init__()
        # torch.manual_seed(1234567)
        self.activation = nn.LeakyReLU() #nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        
        self.lin1 = torch.nn.Linear(ch_input, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin5 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin6 = torch.nn.Linear(hidden_channels, ch_output)

    def forward(self, g, in_feat):
        
        x = self.activation(self.lin1(in_feat));
        x = self.activation(self.lin2(x));
        x = self.activation(self.lin3(x));
        x = self.activation(self.lin4(x));
        x = self.activation(self.lin5(x));
        x = self.lin6(x);
        
        return x
    
## Graph convolutional network =============================
class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats):
        super(GCN, self).__init__()
        self.activation = nn.LeakyReLU() #nn.LeakyReLU() #nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)
        self.conv4 = GraphConv(h_feats, h_feats)
        self.conv5 = GraphConv(h_feats, h_feats)
        # self.conv6 = GraphConv(h_feats, h_feats)
        # self.lin1 = torch.nn.Linear(h_feats, h_feats)
        # self.lin2 = torch.nn.Linear(h_feats, h_feats)
        # self.lin3 = torch.nn.Linear(h_feats, h_feats)
        # self.lin4 = torch.nn.Linear(h_feats, h_feats)
        self.lin5 = torch.nn.Linear(h_feats, num_classes)
        # self.device = device
    
    def forward(self, g, in_feat):
        h = self.activation(self.conv1(g, in_feat))
        h = self.activation(self.conv2(g, h))
        h = self.activation(self.conv3(g, h))
        h = self.activation(self.conv4(g, h))
        h = self.activation(self.conv5(g, h))
        # h = self.activation(self.conv6(g, h))
        # h = self.activation(self.conv3(g, h))
        # h = self.activation(self.lin1(h));
        # h = self.activation(self.lin2(h));
        # h = self.activation(self.lin3(h));
        # h = self.activation(self.lin4(h));
        h = self.lin5(h);

        return h
    
## Graph attention network =============================
class GAT(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats):
        super(GAT, self).__init__()
        self.activation = nn.LeakyReLU() #nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        self.conv1 = GATConv(in_feats, h_feats, num_heads=3)
        self.conv2 = GATConv(h_feats, h_feats, num_heads=3)        
        self.conv3 = GATConv(h_feats, h_feats, num_heads=3)
        self.conv4 = GATConv(h_feats, h_feats, num_heads=3)
        self.conv5 = GATConv(h_feats, h_feats, num_heads=3)
        
        # self.lin1 = torch.nn.Linear(h_feats, h_feats)
        # self.lin2 = torch.nn.Linear(h_feats, h_feats)
        # self.lin3 = torch.nn.Linear(h_feats, h_feats)
        # self.lin4 = torch.nn.Linear(h_feats, h_feats)
        self.lin5 = torch.nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.activation(self.conv1(g, in_feat))
        h = torch.mean(h, dim = 1)
        h = self.activation(self.conv2(g, h))
        h = torch.mean(h, dim = 1)
        h = self.activation(self.conv3(g, h))
        h = torch.mean(h, dim = 1)
        h = self.activation(self.conv4(g, h))
        h = torch.mean(h, dim = 1)
        h = self.activation(self.conv5(g, h))
        h = torch.mean(h, dim = 1)
        # h = self.activation(self.lin1(h));
        # h = self.activation(self.lin2(h));
        # h = self.activation(self.lin3(h));
        # h = self.activation(self.lin4(h));
        h = self.lin5(h);

        return h
    
## GraphSAGE (SAmple and aggreGatE) =============================
class SAGE(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats):
        super(SAGE, self).__init__()
        self.activation = nn.LeakyReLU() #nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        self.conv1 = SAGEConv(in_feats, h_feats, 'pool')
        self.conv2 = SAGEConv(h_feats, h_feats, 'pool')
        self.lin1 = torch.nn.Linear(h_feats, h_feats)
        self.lin2 = torch.nn.Linear(h_feats, h_feats)
        self.lin3 = torch.nn.Linear(h_feats, h_feats)
        self.lin4 = torch.nn.Linear(h_feats, h_feats)
        self.lin5 = torch.nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.activation(self.conv1(g, in_feat))
        h = self.activation(self.conv2(g, h))
        h = self.activation(self.lin1(h));
        h = self.activation(self.lin2(h));
        h = self.activation(self.lin3(h));
        h = self.activation(self.lin4(h));
        h = self.lin5(h);

        return h
    
## Chebyshev Spectral Graph Convolution network =============================
class ChebGCN(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats):
        super(ChebGCN, self).__init__()
        self.activation = nn.LeakyReLU() #nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        self.conv1 = ChebConv(in_feats, h_feats, 5)
        self.conv2 = ChebConv(h_feats, h_feats, 5)
        self.conv3 = ChebConv(h_feats, h_feats, 5)
        self.conv4 = ChebConv(h_feats, h_feats, 5)
        self.conv5 = ChebConv(h_feats, h_feats, 5)
        # self.lin1 = torch.nn.Linear(h_feats, h_feats)
        # self.lin2 = torch.nn.Linear(h_feats, h_feats)
        # self.lin3 = torch.nn.Linear(h_feats, h_feats)
        # self.lin4 = torch.nn.Linear(h_feats, h_feats)
        self.lin5 = torch.nn.Linear(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.activation(self.conv1(g, in_feat))
        h = self.activation(self.conv2(g, h))
        h = self.activation(self.conv3(g, h))
        h = self.activation(self.conv4(g, h))
        h = self.activation(self.conv5(g, h))
        # h = self.activation(self.lin1(h));
        # h = self.activation(self.lin2(h));
        # h = self.activation(self.lin3(h));
        # h = self.activation(self.lin4(h));
        h = self.lin5(h);

        return h

## Equivariant Graph convolutional network =============================
class EGCN2(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats, edge_feat_size=0):
        super(EGCN2, self).__init__()
        self.activation = nn.LeakyReLU() #nn.LeakyReLU() #nn.ReLU() #nn.LeakyReLU(negative_slope=0.01) #nn.Tanh()
        self.conv1 = EGNNConv(in_feats, h_feats, h_feats, edge_feat_size)
        self.conv2 = EGNNConv(h_feats, h_feats, h_feats, edge_feat_size)
        self.conv3 = EGNNConv(h_feats, h_feats, h_feats, edge_feat_size)
        self.conv4 = EGNNConv(h_feats, h_feats, h_feats, edge_feat_size)
        self.conv5 = EGNNConv(h_feats, h_feats, h_feats, edge_feat_size)
        # self.conv6 = GraphConv(h_feats, h_feats)
        # self.lin1 = torch.nn.Linear(h_feats, h_feats)
        # self.lin2 = torch.nn.Linear(h_feats, h_feats)
        # self.lin3 = torch.nn.Linear(h_feats, h_feats)
        # self.lin4 = torch.nn.Linear(h_feats, h_feats)
        self.linh = torch.nn.Linear(h_feats, num_classes)
        self.linx = torch.nn.Linear(h_feats, num_classes)
        # self.device = device
    
    def forward(self, g, in_feat, coord_feat, edge_feat=None):
        h, x = self.conv1(g, in_feat, coord_feat, edge_feat)
        h = self.activation(h); x = self.activation(x);
        h, x = self.conv2(g, h, x, edge_feat)
        h = self.activation(h); x = self.activation(x);
        h, x = self.conv3(g, h, x, edge_feat)
        h = self.activation(h); x = self.activation(x);
        h, x = self.conv4(g, h, x, edge_feat)
        h = self.activation(h); x = self.activation(x);
        h, x = self.conv5(g, h, x, edge_feat)
        h = self.activation(h); x = self.activation(x);
        # h = self.activation(self.conv3(g, h))
        # h = self.activation(self.lin1(h));
        # h = self.activation(self.lin2(h));
        # h = self.activation(self.lin3(h));
        # h = self.activation(self.lin4(h));
        h = self.linh(h)
        # x = self.linx(x)
        # out = torch.cat([h, x], dim=1)

        return h
    
class EGCN(nn.Module):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> node_feat, coord_feat, edge_feat = th.ones(6, 10), th.ones(6, 3), th.ones(6, 2)
    >>> conv = EGNNConv(10, 10, 10, 2)
    >>> h, x = conv(g, node_feat, coord_feat, edge_feat)
    """
    def __init__(self, in_size, out_size, hidden_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.LeakyReLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False)
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src['h'], edges.dst['h'], edges.data['radial'], edges.data['a']],
                dim=-1
            )
        else:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['radial']], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data['x_diff']

        return {'msg_x': msg_x, 'msg_h': msg_h}
    
    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('x', 'x', 'x_diff'))
            graph.edata['radial'] = graph.edata['x_diff'].square().sum(dim=1).unsqueeze(-1)
            # normalize coordinate difference
            graph.edata['x_diff'] = graph.edata['x_diff'] / (graph.edata['radial'].sqrt() + 1e-30)
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e('msg_x', 'm'), fn.mean('m', 'x_neigh'))
            graph.update_all(fn.copy_e('msg_h', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']

            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )
            x = coord_feat + x_neigh
            out = torch.cat([h, x], dim=1)

            return out
    