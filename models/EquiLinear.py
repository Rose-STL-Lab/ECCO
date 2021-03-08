import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# rho1 --> rho1    
class EquiLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(EquiLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, field_feat):
        """
        inputs:
        @field_feat: [batch, num_part, in_feat, 2]
        
        output:
        [batch, num_part, out_feat, 2]
        """
        return self.linear(field_feat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    
# Reg --> Reg    
class EquiLinearRegToReg(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(EquiLinearRegToReg, self).__init__()
        self.k = k
        self.weights = nn.parameter.Parameter(torch.rand(in_features, out_features, k) / in_features)
        # print(self.weights)
        # kernel = self.update_kernel()
        # print(self.kernel)
    
    def update_kernel(self):
            # i or -i ???   stack -2 or -1 ???   torch.flip ???
        return torch.stack([torch.roll(self.weights, i, 2) for i in range(0,self.k)],-1)
        
    def forward(self, field_feat):
        """
        inputs:
        k: int -- number of slices of the circle for regular rep
        @field_feat: [batch, num_part, in_feat, k]
        kernel: [in_feat, out_feat, k, k]
        
        f*k(\theta) = \sum_\psi K(\psi)f(\theta - \psi)

        output:
        [batch, num_part, out_feat, k]
        """
        # x or y ??? 
        kernel = self.update_kernel()
        return torch.einsum('ijyx,...ix->...jy',kernel,field_feat)
    
    
# Rho1 --> Reg
class EquiLinearRho1ToReg(nn.Module):
    def __init__(self, k):
        super(EquiLinearRho1ToReg, self).__init__()
        self.k = k
        SinVec = torch.tensor([math.sin(i * 2 * math.pi / self.k) for i in range(k)],requires_grad=False)
        CosVec = torch.tensor([math.cos(i * 2 * math.pi / self.k) for i in range(k)],requires_grad=False)
        Rho1ToReg = torch.stack([CosVec,SinVec],1)  #[k,2]
        self.register_buffer('Rho1ToReg', Rho1ToReg)
    
    def forward(self, field_feat):
        """
            k: int -- number of slices of the circle for regular rep
            inputs:
                @field_feat: [batch, num_part, in_feat, 2]
            output: [batch, num_part, in_feat, k]
        
            (a,b) --> a Sin + b Cos
        """
        return torch.einsum('yx,...x->...y',self.Rho1ToReg, field_feat)
    
    
# Reg --> Rho1
class EquiLinearRegToRho1(nn.Module):
    def __init__(self, k):
        super(EquiLinearRegToRho1, self).__init__()
        self.k = k
        SinVec = torch.tensor([math.sin(i * 2 * math.pi / self.k) for i in range(k)],requires_grad=False)
        CosVec = torch.tensor([math.cos(i * 2 * math.pi / self.k) for i in range(k)],requires_grad=False)
        RegToRho1 = torch.stack([CosVec,SinVec],0)  #[2,k]
        self.register_buffer('RegToRho1', RegToRho1)
    
    def forward(self, field_feat):
        '''
           k: int -- number of slices of the circle for regular rep
           inputs:
               @field_feat: [batch, num_part, in_feat, k]
           output: 
               retval:      [batch, num_part, in_feat, 2]

           f is a function on circle divided into k parts
           f(i) means f(2\pi *i /k) 
           f --> ( \sum_{i=0}^k  ( f(i) cos(2 \pi i /k) , f(i) sin(2 \pi i /k) )
           This is a fourier transform. 
        '''
        return torch.einsum('yx,...x->...y',self.RegToRho1, field_feat)
