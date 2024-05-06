import torch
import torch.nn as nn
import torch.nn.functional as F
from args import arguments
from util.masking import mask_lane_noise_speeds
from McgVAE.Imputer import LaneSpeedImputer
from McgVAE.Graph_Attention import DynamicGraphConvNet

args = arguments()


class ResBlock (nn.Module):

    def __init__(self, input_shape, dropout, ff_dim):
        super (ResBlock, self).__init__ ()

        # Temporal Linear
        self.norm1 = nn.BatchNorm1d (input_shape[0] * input_shape[1])
        self.linear1 = nn.Linear (input_shape[0], input_shape[0])
        self.dropout1 = nn.Dropout (dropout)

        # Feature Linear
        self.norm2 = nn.BatchNorm1d (input_shape[0] * input_shape[1])
        self.linear2 = nn.Linear (input_shape[-1], ff_dim)
        self.dropout2 = nn.Dropout (dropout)

        self.linear3 = nn.Linear (ff_dim, input_shape[-1])
        self.dropout3 = nn.Dropout (dropout)

    def forward(self, x):
        inputs = x

        # Temporal Linear
        x = self.norm1 (torch.flatten (x, 1, -1)).reshape (x.shape)
        x = torch.transpose (x, 1, 2)
        x = F.relu (self.linear1 (x))
        x = torch.transpose (x, 1, 2)
        x = self.dropout1 (x)

        res = x + inputs

        # Feature Linear
        x = self.norm2 (torch.flatten (res, 1, -1)).reshape (res.shape)
        x = F.relu (self.linear2 (x))
        x = self.dropout2 (x)

        x = self.linear3 (x)
        x = self.dropout3 (x)

        return x + res

# https://github.com/ts-kim/RevIN/blob/master/RevIN.py
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps*self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class Encoder_Lane (nn.Module):
    def __init__(self, seq_len, num_nodes, input_dim, hidden_dim,n_block=2,dropout=0.05,ff_dim=64, target_slice=slice(0,None,None)):
        super (Encoder_Lane, self).__init__ ()
        self.fc1 = nn.Linear (seq_len * num_nodes * input_dim, hidden_dim)
        self.fc_mu = nn.Linear (hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear (hidden_dim, hidden_dim)
        self.rev_norm = RevIN (num_nodes)
        self.linear = nn.Linear (args.seq_len, args.horizon)
        self.target_slice = target_slice
        self.rev_norm = RevIN (num_nodes)
        self.graph_based = args.graph_based
        self.DynamicGraphConvNet = DynamicGraphConvNet (input_dim, args.horizon, num_nodes)
        input_shape = (seq_len, num_nodes)


        self.res_blocks = nn.ModuleList ([ResBlock (input_shape, dropout, ff_dim) for _ in range (n_block)])

    def forward(self, x):   #shape[]
        for res_block in self.res_blocks:
            x = res_block (x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        # x = torch.transpose (x, 1, 2)       #shape[batch_size,seq_len,num_node]
        x = torch.unsqueeze(x,2)
        x = torch.unsqueeze (x,4)
        if self.graph_based:
            x = self.DynamicGraphConvNet(x)
        h = F.relu (self.fc1 (x.view (x.size (0), -1)))
        return self.fc_mu (h), self.fc_logvar (h)  # Returns mu and log(variance)


class Decoder_Lane (nn.Module):
    def __init__(self, seq_len, num_nodes, output_dim, hidden_dim, target_slice=slice(0,None,None)):
        super (Decoder_Lane, self).__init__ ()
        self.fc1 = nn.Linear (hidden_dim, hidden_dim * num_nodes * output_dim)
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rev_norm = RevIN (num_nodes)
        self.target_slice = target_slice

    def forward(self, z):
        h = F.relu (self.fc1 (z))
        h = h.view (-1, self.num_nodes, self.hidden_dim * self.output_dim)
        h = torch.transpose (h, 1, 2) #shape[batch_size,num_node,horizon]
        return h

class Encoder_Road (nn.Module):
    def __init__(self, seq_len, num_nodes, input_dim, hidden_dim,n_block=2,dropout=0.05,ff_dim=64, target_slice=slice(0,None,None)):
        super (Encoder_Road, self).__init__ ()
        self.fc1 = nn.Linear (seq_len * num_nodes * input_dim, hidden_dim)
        self.fc_mu = nn.Linear (hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear (hidden_dim, hidden_dim)
        self.rev_norm = RevIN (num_nodes)
        self.linear = nn.Linear (args.seq_len, args.horizon)
        self.target_slice = target_slice
        self.graph_based = args.graph_based
        self.rev_norm = RevIN (num_nodes)
        self.DynamicGraphConvNet = DynamicGraphConvNet (input_dim, args.horizon, num_nodes)
        input_shape = (seq_len, num_nodes)


        self.res_blocks = nn.ModuleList ([ResBlock (input_shape, dropout, ff_dim) for _ in range (n_block)])

    def forward(self, x):   #shape[]
        for res_block in self.res_blocks:
            x = res_block (x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        # x = torch.transpose (x, 1, 2)       #shape[batch_size,seq_len,num_node]
        x = torch.unsqueeze(x,2)
        x = torch.unsqueeze (x,4)
        if self.graph_based:
            x = self.DynamicGraphConvNet(x)
        h = F.relu (self.fc1 (x.view (x.size (0), -1)))
        return self.fc_mu (h), self.fc_logvar (h)  # Returns mu and log(variance)


class Decoder_Road (nn.Module):
    def __init__(self, seq_len, num_nodes, output_dim, hidden_dim, target_slice=slice(0,None,None)):
        super (Decoder_Road, self).__init__ ()
        self.fc1 = nn.Linear (hidden_dim, hidden_dim * num_nodes * output_dim)
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rev_norm = RevIN (num_nodes)
        self.target_slice = target_slice

    def forward(self, z):
        h = F.relu (self.fc1 (z))
        h = h.view (-1, self.num_nodes, self.hidden_dim * self.output_dim)
        h = torch.transpose (h, 1, 2) #shape[batch_size,num_node,horizon]
        return h


class McgVAE (nn.Module):
    def __init__(self, seq_len, num_road_nodes, num_lane_nodes, input_dim, output_dim, hidden_dim,n_block=2,dropout=0.05,ff_dim=64, target_slice=slice(0,None,None)):
        super (McgVAE, self).__init__ ()
        self.road_encoder = Encoder_Road (seq_len, num_road_nodes, input_dim, hidden_dim)
        self.road_decoder = Decoder_Road (seq_len, num_road_nodes, output_dim, hidden_dim)
        self.lane_encoder = Encoder_Lane (seq_len, num_lane_nodes, input_dim, hidden_dim)
        self.lane_decoder = Decoder_Lane (seq_len, num_lane_nodes, output_dim, hidden_dim)

        input_shape_road = (seq_len, num_road_nodes)
        input_shape_lane = (seq_len, num_lane_nodes)
        self.target_slice = target_slice

        self.rev_norm_road = RevIN (input_shape_road[-1])
        self.rev_norm_lane = RevIN (input_shape_lane[-1])


        self.res_blocks = nn.ModuleList ([ResBlock (input_shape_road, dropout, ff_dim) for _ in range (n_block)])

        self.linear = nn.Linear (args.seq_len, args.horizon)

        self.Imputer = LaneSpeedImputer(seq_len,num_lane_nodes,num_road_nodes,hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp (0.5 * logvar)
        eps = torch.randn_like (std)
        return mu + eps * std

    def forward(self, road_input, lane_input):   #shape[batch_size,seq_len,i_dim,num_node,o_dim]
        road_input = road_input.squeeze ()       #shape[batch_size,seq_len,num_node]
        lane_input = lane_input.squeeze ()
        lane_input = mask_lane_noise_speeds (lane_input, road_input, args.road_lane_count, args.Correlation_threshold)
        lane_input = self.Imputer(lane_input, road_input, args.road_lane_count)
        road_input = self.rev_norm_road (road_input, 'norm')  #shape[batch_size,seq_len,num_node]
        lane_input = self.rev_norm_lane (lane_input, 'norm')

        road_mu, road_logvar = self.road_encoder (road_input)
        lane_mu, lane_logvar = self.lane_encoder (lane_input)   #[batch_size,horizon]

        road_z = self.reparameterize (road_mu, road_logvar)
        lane_z = self.reparameterize (lane_mu, lane_logvar)    #[batch_size,horizon]

        road_recon = self.road_decoder (road_z)
        lane_recon = self.lane_decoder (lane_z)     #[batch_size,num_node, horizon]
        road_recon = self.rev_norm_road (road_recon, 'denorm', self.target_slice)  # shape[batch_size,horizon,num_node]
        road_recon = torch.transpose (road_recon, 1, 2)  ##shape[batch_size,num_node,horizon]
        lane_recon  = self.rev_norm_lane (lane_recon , 'denorm', self.target_slice)  # shape[batch_size,horizon,num_node]
        lane_recon = torch.transpose (lane_recon, 1, 2)

        return road_recon, lane_recon


