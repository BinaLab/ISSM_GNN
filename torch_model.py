import numpy as np

import torch    
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros

### LOSS FUNCTIONS #####################################################################
class vel_loss(nn.Module):
    def __init__(self):
        super(vel_loss, self).__init__();

    def forward(self, obs, prd):
        u_o = obs[:, 0, :, :]; v_o = obs[:, 1, :, :]
        u_p = prd[:, 0, :, :]; v_p = prd[:, 1, :, :]
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        # theta = (u_o*u_p+v_o*v_p)/(vel_o*vel_p)
        # theta = (1 - theta**2)**0.5
        # theta = torch.where(theta >= 0, theta, 0)
        # err_theta = torch.abs(theta)

        err_u = torch.abs(u_o - u_p)
        err_v = torch.abs(v_o - v_p)
        err_vel = torch.abs(vel_o - vel_p)        

        err_sum = torch.mean((err_u + err_v + err_vel))*100
        # err_sum += torch.nanmean(err_theta)/10000
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        return err_sum  
    
class single_loss(nn.Module):
    def __init__(self, landmask):
        super(single_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd):
        n_outputs = obs.size()[1]
        err_sum = 0
        for i in range(0, n_outputs):
            err = torch.square(obs[:, i, :, :] - prd[:, i, :, :])
            # err = torch.mean(err, dim=0)[self.landmask == 0]
            err_sum += torch.mean(err)**0.5*100
        return err_sum

class custom_loss(nn.Module):
    def __init__(self, landmask):
        super(custom_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd):
        sic = prd[:, 2, :, :]
        u_o = obs[:, 0, :, :]; v_o = obs[:, 1, :, :]
        u_p = prd[:, 0, :, :]; v_p = prd[:, 1, :, :]
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        theta = torch.acos((u_o*u_p+v_o*v_p)/(vel_o*vel_p))
        theta = torch.where(torch.isnan(theta), 0, theta)

        err_u = torch.square(u_o - u_p) #[sic > 0]
        err_v = torch.square(v_o - v_p) #[sic > 0]
        err_vel = torch.abs(vel_o - vel_p) #[sic > 0]
        err_theta = torch.abs(theta)
        
        err1 = torch.mean(err_u + err_v, dim=0)[torch.where(self.landmask == 0)]
        err_sum = torch.mean(err1)*1000 

        err_sic = torch.square(obs[:, 2, :, :]-prd[:, 2, :, :])
        
        neg_sic = torch.where(prd[:, 2, :, :] < 0, abs(prd[:, 2, :, :]), 0)
        err2 = torch.mean(err_sic, dim=0)[torch.where(self.landmask == 0)]
        err_sum += torch.mean(err2)*1000
        
        if obs.size()[1] > 3:
            err_sit = torch.abs(obs[:, 3, :, :]-prd[:, 3, :, :])  
            neg_sit = torch.where(prd[:, 3, :, :] < 0, abs(prd[:, 3, :, :]), 0)
            err3 = torch.mean(err_sit, dim=0)[torch.where(self.landmask == 0)]   
            err_sum += torch.mean(err3)*5000
        
        # err_sum += torch.mean(err_sic + err_sit)*100
        # err_sum += torch.nanmean(err_theta)*0.5/3.141592
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        return err_sum   

def corrcoef(x, y):
    x = x.flatten()
    y = y.flatten()
    xm = torch.mean(x)
    ym = torch.mean(y)

    r1 = torch.sum((x-xm)*(y-ym))
    r2 = torch.sum(torch.square(x-xm))*torch.sum(torch.square(y-ym))
    r = r1/(r2**0.5)
    return r
    
class physics_loss(nn.Module):
    def __init__(self, landmask):
        super(physics_loss, self).__init__();
        self.landmask = landmask

    def forward(self, obs, prd, sic0):
        
        sic_p = prd[:, 2, :, :]
        sic_o = obs[:, 2, :, :]
        u_o = obs[:, 0, :, :]*50; v_o = obs[:, 1, :, :]*50
        u_p = prd[:, 0, :, :]*50; v_p = prd[:, 1, :, :]*50  
        
        vel_o = (u_o**2 + v_o**2)**0.5
        vel_p = (u_p**2 + v_p**2)**0.5
        
        err_u = torch.square(u_o - u_p) #[sic > 0]
        err_v = torch.square(v_o - v_p) #[sic > 0]
        
        sicmask = torch.max(sic_o, dim=0)[0]
        err1 = torch.mean(err_u + err_v, dim=0)[torch.where((self.landmask == 0) & (sicmask > 0))]
        err_sum = torch.mean(err1)*10

        err_sic = torch.square(sic_o - sic_p)*500
        
        neg_sic = torch.where(sic_p < 0, abs(sic_p), 0)
        pos_sic = torch.where(sic_p > 1, abs(sic_p-1), 0)
        
        err2 = torch.mean(err_sic + neg_sic + pos_sic, dim=0)[torch.where(self.landmask == 0)]
        err_sum += torch.mean(err2)
        
        if obs.size()[1] > 3:
            sit_p = prd[:, 3, :, :]
            sit_o = obs[:, 3, :, :]
            err_sit = torch.square(sit_o - sit_p)
            neg_sit = torch.where(sit_p < 0, abs(sit_p), 0)
            err3 = torch.mean(err_sit + neg_sit, dim=0)[torch.where(self.landmask == 0)]   
            err_sum += torch.mean(err3)*50
        
        # physics loss ===============================================
        # advection
        dx = (sic_p[:, 1:-1, 2:]-sic_p[:, 1:-1, :-2]) + (sic_p[:, 2:, 2:]-sic_p[:, 2:, :-2]) + (sic_p[:, :-2, 2:]-sic_p[:, :-2, :-2])
        dy = (sic_p[:, 2:, 1:-1]-sic_p[:, :-2, 1:-1]) + (sic_p[:, 2:, 2:]-sic_p[:, :-2, 2:]) + (sic_p[:, 2:, :-2]-sic_p[:, :-2, :-2])    
        advc = (u_p[:, 1:-1, 1:-1]*dx/3 + v_p[:, 1:-1, 1:-1]*dy/3)/25
        
        # divergence
        dx = (u_p[:, 1:-1, 2:]-u_p[:, 1:-1, :-2]) + (u_p[:, 2:, 2:]-u_p[:, 2:, :-2]) + (u_p[:, :-2, 2:]-u_p[:, :-2, :-2])
        dy = (v_p[:, 1:-1, 2:]-v_p[:, 1:-1, :-2]) + (v_p[:, 2:, 2:]-v_p[:, 2:, :-2]) + (v_p[:, :-2, 2:]-v_p[:, :-2, :-2])
        divc = dx/3 + dy/3
        divc = divc*sic_p[:, 1:-1, 1:-1]/25
        
        dsic = sic_p[:, 1:-1, 1:-1] - sic0[:, 1:-1, 1:-1]
        
        residual = dsic + advc
        r = corrcoef(dsic, advc)
        
        # SIC change
        err_phy = 0
        err_phy += torch.mean(torch.where(abs(residual) > 2, abs(residual)-2, 0))
        if r > 0:
            err_phy += r
        # err_phy = torch.mean(torch.where((div > 0) & (d_sic > 0), err_u + err_v + err_sic, 0))
        
        w = torch.tensor(1.0)
        err_sum += w*err_phy
        
        return err_sum    
    
    
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, obs, sid, sic, sit):
        
        # Sea ice drift error
        err_u = torch.abs(obs[:, 0, :, :] - sid[:, 0, :, :])
        err_v = torch.abs(obs[:, 1, :, :] - sid[:, 1, :, :])
        loss0 = torch.mean((err_u + err_v))*100 
        
        # SIC error
        err_sic = torch.abs(obs[:, 2, :, :]-sic)
        loss1 = torch.mean(err_sic)*100
        
        # SIT error
        err_sit = torch.abs(obs[:, 3, :, :]-sit)
        loss2 = torch.mean(err_sit)*100

#         precision0 = torch.exp(-self.log_vars[0])
#         loss0 = precision0*loss0 + self.log_vars[0]

#         precision1 = torch.exp(-self.log_vars[1])
#         loss1 = precision1*loss1 + self.log_vars[1]

#         precision2 = torch.exp(-self.log_vars[2])
#         loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2
    
### MAKE INPUT DATASETS #########################################################
def convert_cnn_input2D(data_input, data_output, days, months, years, dayint = 3, forecast = 3, exact = False):
    # dayint: days before forecast (use as input features)
    # forecast: lead day for forecasting (output features)
    # exact: if True, only that exact date is forecasted; if False, all days before the lead day is forecasted
    # Input & output should be entire images for CNN
    
    # Cehck sequential days
    seq_days = []
    step = 0

    for i in range(0, len(days)):
        if (days[i] ==1) & (years[i] != years[0]):
            step += days[i-1]
        seq_days.append(days[i] + step)

    seq_days = np.array(seq_days)
    
    n_samples, row, col, var_ip = np.shape(data_input)
    _, _, _, var_op = np.shape(data_output)

    cnn_input = np.zeros([n_samples, row, col, var_ip * dayint], dtype = np.float16)
    if exact:
        cnn_output = np.zeros([n_samples, row, col, var_op], dtype = np.float16)
    else:
        cnn_output = np.zeros([n_samples, row, col, var_op * forecast], dtype = np.float16)
    valid = []
    
    for n in range(dayint-1, n_samples-forecast):
        if seq_days[n+forecast] - seq_days[n-dayint+1] == dayint + forecast-1:
            valid.append(n)
            for i in range(0, dayint):
                for v in range(0, var_ip):            
                    cnn_input[n, :, :, v+i*var_ip] = (data_input[n-i, :, :, v]).astype(np.float16)
            # if v in range(0, var_op):
            if exact:
                cnn_output[n, :, :, :] = (data_output[n+forecast-1, :, :, :]).astype(np.float16)
            else:
                for j in range(0, forecast):
                    for v in range(0, var_op):            
                        cnn_output[n, :, :, v+j*var_op] = (data_output[n+j, :, :, v]).astype(np.float16)
                
                
    return cnn_input[valid, :, :, :], cnn_output[valid, :, :, :], days[valid], months[valid], years[valid]

### ML MODELS #####################################################################
class FC(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(n_inputs, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        return x

class linear_regression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, row, col):
        super(linear_regression, self).__init__()        
        self.asiu = torch.nn.Parameter(torch.ones(1, inputSize, row, col)*0.5)
        self.bsiu = torch.nn.Parameter(torch.ones(1, row, col)*0.5)
        self.asiv = torch.nn.Parameter(torch.ones(1, inputSize, row, col)*0.5)
        self.bsiv = torch.nn.Parameter(torch.ones(1, row, col)*0.5)
        self.asic = torch.nn.Parameter(torch.ones(1, inputSize, row, col)*0.5)
        self.bsic = torch.nn.Parameter(torch.ones(1, row, col)*0.5)

    def forward(self, x):
        siu = torch.sum(self.asiu*x, dim=1) + self.bsiu
        siv = torch.sum(self.asiv*x, dim=1) + self.bsiv
        sic = torch.sum(self.asic*x, dim=1) + self.bsic
        out = torch.cat([siu.unsqueeze(1), siv.unsqueeze(1), sic.unsqueeze(1)], dim=1)
        return out

# CNN model
class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_filters=32, kernel = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_filters, kernel, padding = "same")
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv5 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv6 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv7 = nn.Conv2d(n_filters, n_filters, kernel, padding = "same")
        self.conv8 = nn.Conv2d(n_filters, n_outputs, kernel, padding = "same")

    def forward(self, x):
        # x = F.tanh(self.conv1(x)) #F.leaky_relu(self.conv1(x))
        # x = F.tanh(self.conv2(x)) #F.leaky_relu(self.conv2(x))
        # x = F.tanh(self.conv3(x)) #F.leaky_relu(self.conv3(x))
        # x = F.tanh(self.conv4(x)) #F.leaky_relu(self.conv4(x))
        # x = F.tanh(self.conv5(x)) #F.leaky_relu(self.conv5(x))
        # x = F.tanh(self.conv6(x)) #F.leaky_relu(self.conv6(x))
        # x = F.tanh(self.conv7(x)) #F.leaky_relu(self.conv7(x))
        # x = F.tanh(self.conv8(x)) #F.leaky_relu(self.conv8(x))
        
        x = F.leaky_relu(self.conv1(x), negative_slope=1)
        x = F.leaky_relu(self.conv2(x), negative_slope=1)
        x = F.leaky_relu(self.conv3(x), negative_slope=1)
        x = F.leaky_relu(self.conv4(x), negative_slope=1)
        x = F.leaky_relu(self.conv5(x), negative_slope=1)
        x = F.leaky_relu(self.conv6(x), negative_slope=1)
        x = F.leaky_relu(self.conv7(x), negative_slope=1)
        x = F.leaky_relu(self.conv8(x), negative_slope=1)
        
        # x = F.linear(self.conv1(x), weight = 1)
        # x = F.linear(self.conv2(x), weight = 1)
        # x = F.linear(self.conv3(x), weight = 1)
        # x = F.linear(self.conv4(x), weight = 1)
        # x = F.linear(self.conv5(x), weight = 1)
        # x = F.linear(self.conv6(x), weight = 1)
        # x = F.linear(self.conv7(x), weight = 1)
        # x = F.linear(self.conv8(x), weight = 1)
        
        return x

    
    
class GCNet(torch.nn.Module):
    def __init__(self, ch_input, ch_output, hidden_channels = 128):
        super().__init__()
        # torch.manual_seed(1234567)
        self.activation = nn.Tanh()
        self.conv1 = GCNConv(ch_input, hidden_channels, improved=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, ch_output)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index):
        x = self.activation(self.conv1(x, edge_index)); #self.conv1(x)
        x = self.activation(self.conv2(x, edge_index));
        x = self.dropout(self.activation(self.lin1(x)));
        x = self.dropout(self.activation(self.lin2(x)));
        x = self.dropout(self.activation(self.lin3(x)));
        x = self.lin4(x);
        
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
    def __init__(self, ch_input, ch_output, hidden_channels = 128):
        super().__init__()
        # torch.manual_seed(1234567)
        self.activation = nn.Tanh()
        self.emb = nn.Linear(ch_input, hidden_channels) 
        self.gnn = ConvEGNN(hidden_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)
        
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(0.25),
            nn.Tanh(),
            nn.Linear(hidden_channels, ch_output)
        )

    def forward(self, x, edge_index):
        x.pos = x[:, :2]
        x.h = x
        x.h = self.emb(x.h)
        x = self.gnn(x, edge_index); #self.conv1(x)
        # x = self.activation(self.conv2(x, edge_index));
        x1 = self.lin(x.h);
        
        return x1
    
class ConvEGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, cuda=False):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # preducts "soft" edges based on messages 
        self.f_inf = nn.Sequential( 
            nn.Linear(hid_dim, 1),
            nn.Sigmoid())
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, b, edge_index):
        e_st = edge_index[0, :]
        e_end = edge_index[1, :]
        
        dists = torch.norm(b.pos[e_st] - b.pos[e_end], dim=1).reshape(-1, 1)
        
        # compute messages
        print(b.h[e_st].size(), b.h[e_end].size(), dists.size())
        tmp = torch.hstack([b.h[e_st], b.h[e_end], dists])
        m_ij = self.f_e(tmp)
        
        # predict edges
        e_ij = self.f_inf(m_ij)
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = index_sum(b.shape[0], e_ij*m_ij, edge_index[0,:], self.cuda)
        
        # update hidden representations
        b.h += self.f_h(torch.hstack([b.h, m_i]))

        return b
    
class NetEGNN(nn.Module):
    def __init__(self, in_dim=15, hid_dim=128, out_dim=1, n_layers=7, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = [ConvEGNN(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers)]
        self.gnn = nn.Sequential(*self.gnn)
        
        self.pre_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
        
        self.post_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, b):
        b.h = self.emb(b.h)
        
        b = self.gnn(b)
        h_nodes = self.pre_mlp(b.h)
        
        # h_graph is num_graphs x hid_dim
        h_graph = index_sum(b.nG, h_nodes, b.batch, self.cuda) 
        
        out = self.post_mlp(h_graph)
        return out

# Convolutional LSTM cell
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs

# UNET model
class UNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, k=3):
        super().__init__()
        
        self.activation = nn.Tanh() #nn.LeakyReLU(0.1) #nn.Tanh() #
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # input: 320x320x3
        self.e11 = nn.Conv2d(n_inputs, 64, kernel_size=k, padding="same") # output: 320x320x32
        self.e12 = nn.Conv2d(64, 64, kernel_size=k, padding="same") # output: 320x320x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 160x160x32

        # input: 160x160x32
        self.e21 = nn.Conv2d(64, 128, kernel_size=k, padding="same") # output: 160x160x64
        self.e22 = nn.Conv2d(128, 128, kernel_size=k, padding="same") # output: 160x160x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 80x80x64

        # input: 80x80x64
        self.e31 = nn.Conv2d(128, 256, kernel_size=k, padding="same") # output: 80x80x128
        self.e32 = nn.Conv2d(256, 256, kernel_size=k, padding="same") # output: 80x80x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 40x40x128

        # input: 40x40x128
        self.e41 = nn.Conv2d(256, 512, kernel_size=k, padding="same") # output: 40x40x256
        self.e42 = nn.Conv2d(512, 512, kernel_size=k, padding="same") # output: 40x40x256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 20x20x256

        # input: 20x20x256
        self.e51 = nn.Conv2d(512, 1024, kernel_size=k, padding="same") # output: 20x20x512
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=k, padding="same") # output: 20x20x512

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=k, padding="same")
        self.d12 = nn.Conv2d(512, 512, kernel_size=k, padding="same")

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=k, padding="same")
        self.d22 = nn.Conv2d(256, 256, kernel_size=k, padding="same")

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=k, padding="same")
        self.d32 = nn.Conv2d(128, 128, kernel_size=k, padding="same")

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=k, padding="same")
        self.d42 = nn.Conv2d(64, 64, kernel_size=k, padding="same")

        # Output layer
        self.outconv = nn.Conv2d(64, n_outputs, kernel_size=k, padding="same")
        # self.sidconv1 = nn.Conv2d(64, 64, kernel_size=k, padding="same")
        # self.sidconv2 = nn.Conv2d(64, 2, kernel_size=k, padding="same")
        # self.sicconv1 = nn.Conv2d(64, 64, kernel_size=k, padding="same")
        # self.sicconv2 = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        # self.sitconv1 = nn.Conv2d(64, 64, kernel_size=k, padding="same")
        # self.sitconv2 = nn.Conv2d(64, 1, kernel_size=k, padding="same")
        
    def forward(self, x):
        # Encoder
        # xe11 = self.activation(self.e11(x))
        xe12 = self.activation(self.e12(self.activation(self.e11(x))))
        xp1 = self.pool1(xe12)

        # xe21 = self.activation(self.e21(xp1))
        xe22 = self.activation(self.e22(self.activation(self.e21(xp1))))
        xp2 = self.pool2(xe22)

        # xe31 = self.activation(self.e31(xp2))
        xe32 = self.activation(self.e32(self.activation(self.e31(xp2))))
        xp3 = self.pool3(xe32)

        # xe41 = self.activation(self.e41(xp3))
        xe42 = self.activation(self.e42(self.activation(self.e41(xp3))))
        xp4 = self.pool4(xe42)

        # xe51 = self.activation(self.e51(xp4))
        xe52 = self.activation(self.e52(self.activation(self.e51(xp4))))
        
        # Decoder
        # xu1 = self.upconv1(xe52)
        xu11 = torch.cat([self.upconv1(xe52), xe42], dim=1)
        # xd11 = self.activation(self.d11(xu11))
        xd12 = self.activation(self.d12(self.activation(self.d11(xu11))))

        # xu2 = self.upconv2(xd12)
        xu22 = torch.cat([self.upconv2(xd12), xe32], dim=1)
        # xd21 = self.activation(self.d21(xu22))
        xd22 = self.activation(self.d22(self.activation(self.d21(xu22))))

        # xu3 = self.upconv3(xd22)
        xu33 = torch.cat([self.upconv3(xd22), xe22], dim=1)
        # xd31 = self.activation(self.d31(xu33))
        xd32 = self.activation(self.d32(self.activation(self.d31(xu33))))

        # xu4 = self.upconv4(xd32)
        xu44 = torch.cat([self.upconv4(xd32), xe12], dim=1)
        # xd41 = self.activation(self.d41(xu44))
        xd42 = self.activation(self.d42(self.activation(self.d41(xu44))))

        # Output layer
        
        # sid = self.sidconv1(xd42)
        # sid = self.sidconv2(sid)
        # sic = self.sicconv1(xd42)
        # sic = self.sicconv2(sic)
        # # sit = self.sitconv1(xd42)
        # # sit = self.sitconv2(sit)        
        # out = torch.cat([sid, sic], dim=1)
        out = self.outconv(xd42)

        return out
    


# Graph convolutional LSTM
class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C