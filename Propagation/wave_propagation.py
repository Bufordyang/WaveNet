from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import sympy as sym
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.sparse as sp



def gen_wave_series(mother_wave, scale, x=sym.symbols('x'), overlap=False):
    '''
    mother_wave: sym.sybol; the original function which need to be modulated
    scale: int; the interval resolution scale, which decide the number of wave bases
    x: sym.symbol; dedicate the character of mother_wave
    overlap: bool; decide if construct overlaped base
    '''
    expr = mother_wave
    wave_set = []
    num_split_interval = 0
    if overlap:
        for i in range(scale-1, -scale, -1):
            wave_set.append(mother_wave.subs(x, scale*x+i))
            num_split_interval += 1
    else:
        for i in range(scale-1, -scale, -2):
            wave_set.append(mother_wave.subs(x, scale*x+i))
            num_split_interval += 1

    return wave_set, num_split_interval



class Wave_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Wave_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K))       
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def Haar(self, x, scale):
        x[x < 0 ] = 0
        start = 2/self.K * scale
        end = start + 2/self.K
        mask = torch.logical_and(x >= start, x < end)  
        if end == 2:
            mask = torch.logical_and(x >= start, x <= end)
        A = torch.where(mask, torch.ones_like(x), torch.zeros_like(x))
        return A

    def forward(self, x, eigen, eigen_vector):
        # TEMP=F.relu(self.temp) 
        TEMP=self.temp
        sum_matrix = []
        # Wavelet MUL-Marr
        Pro_Matrix = TEMP[0] * self.Haar(eigen, scale=0)
        for i in range(1,self.K):
            Pro_Matrix += TEMP[i] * self.Haar(eigen, scale=i)

        Pro_Matrix = eigen_vector @ torch.diag(Pro_Matrix) @ eigen_vector.T
        # coo_matrix = sp.coo_matrix(Pro_Matrix.cpu().detach().numpy())
        # edge_index = np.vstack((coo_matrix.row, coo_matrix.col))
        # x=self.propagate(edge_index,x=x,norm=coo_matrix.data,size=None)
        x = Pro_Matrix @ x
        return x, Pro_Matrix


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

