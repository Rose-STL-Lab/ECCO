import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABCMeta, abstractmethod
from EquiLinear import *


class EquiCtsConvBase(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(EquiCtsConvBase, self).__init__()
    
    @abstractmethod
    def computeKernel(self):
        pass
    
    def GetAttention(self, relative_field):
        r = torch.sum(relative_field ** 2, axis=-1)
        return torch.relu((1 - r) ** 3).unsqueeze(-1)
    
    @classmethod
    def RotMat(cls, theta):
        m = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)], 
                [torch.sin(theta), torch.cos(theta)]
            ], requires_grad=False)
        return m  
        
    @classmethod
    def Rho1RotMat(cls, theta):
        m = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)], 
                [torch.sin(theta), torch.cos(theta)]
            ], requires_grad=False)
        return m
    
    @classmethod
    def RegRotMat(cls, theta, k):
        slice_angle = 2 * math.pi / k
        index_shift = theta / slice_angle
        i = np.floor(index_shift).astype(np.int)
        # divide weights between i and i+1   first_col = [ 0 0 0 ... 0 w_i w_{i+1} 0 0 0 ... 0 0 0] 
        first_col = torch.zeros(k)
        
        offset = (theta - slice_angle * i) / slice_angle
        w_i = 1 - offset
        w_ip = offset
        first_col[np.mod(i,8)], first_col[np.mod(i+1,8)] = w_i, w_ip
        
        m = torch.stack([torch.roll(first_col, i, 0) for i in range(k)], -1)
        #like a permuation matrix which sends i -> i + \theta / ( 2 \pi / k ) 
        #Note if k = num_theta, then it is a permutation matrix 
        return m
    
    def PolarCoords(self, vec, epsilon = 1e-9):
        # vec: [batch, num_m, num_n, pos_dim]
        # Convert to Polar
        r = torch.sqrt(vec[...,0] **2 + vec[...,1] **2 + epsilon) 
         
        cond_nonzero = ~((vec[...,0] == 0.) & (vec[...,1] == 0.))
    
        theta = torch.zeros(vec[...,0].shape, device=self.outer_weights.device)
        theta[cond_nonzero] = torch.atan2(vec[...,1][cond_nonzero], vec[...,0][cond_nonzero])
        
        out = [r, theta]
        out = torch.stack(out, -1)
        return out
    
    def InterpolateKernel(self, kernel, pos):
        """
        @kernel: [c_out, c_in=feat_dim, r, theta, 2, 2] -> [batch, C=c_out*c_in*4, r, theta]
        @pos: [batch, num_m, num_n, 2] -> [batch, num_m, num_n, 2]
        
        return out: [batch, C=c_out*c_in*4, num_m, num_n] -> [batch, num_m, num_n, c_out, c_in, 2, 2]
        """
        # kernel:  [c_out, c_in=feat_dim, r, theta, 2, 2]
        kernels = kernel.permute(0, 1, 4, 5, 2, 3)
        # kernels: [c_out, c_in=feat_dim, 2, 2, r, theta]
        
        kernels = kernels.reshape(-1, *kernels.shape[4:]).unsqueeze(0)
        # kernels: [1, c_out*c_in*2*2, r, theta]
        
        kernels = kernels.expand((pos.shape[0], *kernels.shape[1:]))
        # kernels: [batch_size, c_out*c_in*2*2, r, theta]
                  #[N, C, H, W]
        

        # Copy first and last column to wrap thetas.
        padded_kernels = torch.cat([
            kernels[..., -1].unsqueeze(-1), 
            kernels, 
            kernels[..., 0].unsqueeze(-1)
        ],dim = -1)
        padded_kernels = padded_kernels.permute(0,1,3,2)
        # padded_kernels: [batch, C=c_out*c_in*4, theta+2, r]
        
        
        grid = pos
        # adjust radii [0,1] -> [-1,1]
        grid[...,0] = 2*grid[...,0] - 1
        # adjust angles [-pi,pi] -> [-1,1]
        grid[...,1] *= 1/math.pi
        # shrink thetas slightly to account for padding
        grid[...,1] *= self.num_theta / (self.num_theta + 2)
        # grid [batch, num_m, num_n, 2]
        #      [N, H_out, W_out, 2]
        
        # print("grid",grid)
        # print("padded_kernels_shape [batch_size, c_out*c_in*2*2, theta+2, r]:",padded_kernels.shape)
        #print("kernels",padded_kernels)
        
        out = F.grid_sample(padded_kernels, grid, padding_mode='zeros', 
                            mode='bilinear', align_corners=False)  #bilinear
        # out: [batch, C=c_out*c_in*4, num_m, num_n]
        #      [N, C, H_out, W_out]
        
        out = out.permute(0, 2, 3, 1)
        # out: [batch, num_m, num_n, C=c_out*c_in*4]
        out = out.reshape(*pos.shape[:-1], *kernel.shape[0:2], *kernel.shape[-2:])
        # out: [batch, num_m, num_n, c_out, c_in, 2, 2]     
        return out
    
    def ContinuousConv(
        self, field, center, field_feat, 
        field_mask, ctr_feat=None
    ):
        """
        @kernel: [c_out, c_in=feat_dim, r, theta, 2, 2]
        @field: [batch, num_n, pos_dim=2] -> [batch, 1, num_n, pos_dim]
        @center: [batch, num_m, pos_dim=2] -> [batch, num_m, 1, pos_dim]
        @field_feat: [batch, num_n, c_in=feat_dim, 2] -> [batch, 1, num_n, c_in, 2]
        @ctr_feat: [batch, 1, feat_dim]
        @field_mask: [batch, num_n, 1]
        """
        kernel = self.computeKernel()
        
        relative_field = (field.unsqueeze(1) - center.unsqueeze(2)) / self.radius
        # relative_field: [batch, num_m, num_n, pos_dim]
        

        polar_field = self.PolarCoords(relative_field)
        # polar_field: [batch, num_m, num_n, pos_dim]
        
        kernel_on_field = self.InterpolateKernel(kernel, polar_field)
        # kernel_on_field: [batch, num_m, num_n, c_out, c_in, 2, 2]
        
        if self.use_attention:
            # print(relative_field.shape)
            # print(field_mask.unsqueeze(1).shape)
            attention = self.GetAttention(relative_field) * field_mask.unsqueeze(1)
            # attention: [batch, num_m, num_n, 1]

            if self.normalize_attention:
                psi = torch.sum(attention, axis=2).squeeze(-1)
                psi[psi == 0.] = 1
                psi = psi.unsqueeze(-1).unsqueeze(-1)
            else:
                psi = 1.0
        else:
            attention = torch.ones(*relative_field.shape[0:3],1)
            
            if self.normalize_attention:
                psi = torch.sum(attention, axis=2).squeeze(-1)
                psi[psi == 0.] = 1
                psi = psi.unsqueeze(-1).unsqueeze(-1)
            else:
                psi = 1.0
        
        attention_field_feat = field_feat.unsqueeze(1)*attention.unsqueeze(-1)
        # attention_field_feat: [batch, num_m, num_n, c_in, 2]

        out = torch.einsum('bmnoiyx,bmnix->bmoy', kernel_on_field, attention_field_feat)
        # out: [batch, num_m, c_out, 2]
        
        return out / psi
        
    def forward(
        self, field, center, field_feat, 
        field_mask, ctr_feat=None
    ):
        out = self.ContinuousConv(
            field, center, field_feat, field_mask, 
            ctr_feat
        )
        return out
    

class EquiCtsConv2d(EquiCtsConvBase):
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(EquiCtsConv2d, self).__init__()
        self.num_theta = num_theta
        self.num_radii = num_radii
        
        kernel_basis_outer, kernel_bullseye = self.GenerateKernelBasis(num_radii, num_theta, matrix_dim)
        self.register_buffer('kernel_basis_outer', kernel_basis_outer)
        self.register_buffer('kernel_bullseye', kernel_bullseye)
        
        outer_weights = torch.rand(in_channels, out_channels, num_radii, matrix_dim, matrix_dim)
        outer_weights -= 0.5
        k = 1 / torch.sqrt(torch.tensor(in_channels, dtype=torch.float))
        outer_weights *= 1 * k
        self.outer_weights = torch.nn.parameter.Parameter(outer_weights)
        
        bullseye_weights = torch.rand(in_channels, out_channels)
        bullseye_weights -= 0.5
        bullseye_weights *= 1 * k
        self.bullseye_weights = torch.nn.parameter.Parameter(bullseye_weights)
        
        self.radius = radius

        self.use_attention = use_attention
        self.normalize_attention = normalize_attention
    
    def computeKernel(self):
        # print("[r, d, d, r, theta, d, d] ",self.kernel_basis.shape)
        # print("[c_in,c_out, r, d , d]", self.weights.shape)
        kernel = (torch.einsum('pabrtij,xypab->yxrtij',self.kernel_basis_outer, self.outer_weights) +
                 torch.einsum('rtij,xy->yxrtij',self.kernel_bullseye,self.bullseye_weights))
        return kernel
    
    def GenerateKernelBasis(self, r, theta, matrix_dim=2):
        """
        output: KB : [r+1, d, d, r+1, theta, d, d]  
        KB_bullseye : 
        """ 
        d = matrix_dim
        KB_outer = torch.zeros(r, d, d, r+1, theta, d, d, requires_grad=False)
        K_bullseye = self.GenerateKernelBullseyeElement(r+1, theta, d)
        
        for i in range(d):
            for j in range(d):
                for r1 in range(0, r):
                    KB_outer[r1,i,j] = self.GenerateKernelBasisElement(r+1, theta, i, j, r1+1, d) 
            
        return KB_outer, K_bullseye
    
    def GenerateKernelBasisElement(self, r, theta, i, j, r1, matrix_dim=2):
        """
        output: K: [r, theta, d, d]
        """
        d = matrix_dim
        K = torch.zeros(r, theta, d, d, requires_grad=False)
        K[r1] = self.GenerateKernelBasisElementColumn(theta, i, j, d)
        return K
        
    def GenerateKernelBasisElementColumn(self, theta, i, j, matrix_dim=2):
        # d = matrix_dim
        # 0 <= i,j <= d-1
        # C = kernelcolumn: [theta, d, d]
        # C[0,:,:] = 0
        # C[0,i,j] = 1
        # for k in range(1,theta):
        #   C[k] = RotMat(k*2*pi/theta) * C[0] * RotMat(-k*2*pi/theta) 
        # # K[g v] = g K[v] g^{-1}
        d = matrix_dim
        C = torch.zeros(theta, d, d, requires_grad=False)
        C[0,i,j] = 1
        # TODO: rho 1 -> rho n
        for k in range(1, theta):
            theta_i = torch.tensor(k*2*math.pi/theta)
            C[k] = self.RotMat(theta_i).matmul(C[0]).matmul(self.RotMat(-theta_i))
        return C
    
    def GenerateKernelBullseyeElement(self, r, theta, matrix_dim=2):
        """
        output: K: [r, theta, d, d]
        """
        d = matrix_dim
        K = torch.zeros(r, theta, d, d, requires_grad=False)
        K[0] = self.GenerateKernelBullseyeElementColumn(theta, d)
        return K
    
    def GenerateKernelBullseyeElementColumn(self, theta, matrix_dim=2):
        d = matrix_dim
        C = torch.zeros(theta, d, d, requires_grad=False)
        C[:,0,0] = 1
        C[:,1,1] = 1
        return C
            

class EquiCtsConv2dRegToRho1(EquiCtsConvBase):
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, k, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(EquiCtsConv2dRegToRho1, self).__init__()
        self.num_theta = num_theta
        self.num_radii = num_radii
        self.k = k

        RegToRho1Mat = EquiLinearRegToRho1(k).RegToRho1    #[2,k]
        self.register_buffer('RegToRho1Mat', RegToRho1Mat)
    
        
        kernel_basis_outer, kernel_bullseye = self.GenerateKernelBasis(num_radii, num_theta, matrix_dim)
        self.register_buffer('kernel_basis_outer', kernel_basis_outer)
        self.register_buffer('kernel_bullseye', kernel_bullseye)
        
        outer_weights = torch.rand(in_channels, out_channels, num_radii, matrix_dim, k)
        outer_weights -= 0.5
        scale_norm = 1 / torch.sqrt(torch.tensor(in_channels, dtype=torch.float))
        outer_weights *= 1 * scale_norm
        self.outer_weights = torch.nn.parameter.Parameter(outer_weights)
        
        bullseye_weights = torch.rand(in_channels, out_channels)
        bullseye_weights -= 0.5
        bullseye_weights *= 1 * scale_norm
        self.bullseye_weights = torch.nn.parameter.Parameter(bullseye_weights)
        
        self.radius = radius

        self.use_attention = use_attention
        self.normalize_attention = normalize_attention
        
    def computeKernel(self):
        # print("[r, d, d, r, theta, d, k] ",self.kernel_basis.shape)
        # print("[c_in,c_out, r, d, k]", self.weights.shape)
        kernel = (torch.einsum('pabrtij,xypab->yxrtij',self.kernel_basis_outer, self.outer_weights) +
                 torch.einsum('rtij,xy->yxrtij',self.kernel_bullseye,self.bullseye_weights))
        return kernel
    
    def GenerateKernelBasis(self, r, theta, matrix_dim=2):
        """
        output: KB : [r+1, d, k, r+1, theta, d, k]  
        KB_bullseye : 
        """ 
        d = matrix_dim
        k = self.k
        
        KB_outer = torch.zeros(r, d, k, r+1, theta, d, k, requires_grad=False)
        K_bullseye = self.GenerateKernelBullseyeElement(r+1, theta, d)
        
        for i in range(d):
            for j in range(k):
                for r1 in range(0, r):
                    KB_outer[r1,i,j] = self.GenerateKernelBasisElement(r+1, theta, i, j, r1+1, d) 
            
        return KB_outer, K_bullseye
    
    def GenerateKernelBasisElement(self, r, theta, i, j, r1, matrix_dim=2):
        """
        output: K: [r, theta, d, k]
        """
        d = matrix_dim
        k = self.k
        
        K = torch.zeros(r, theta, d, k, requires_grad=False)
        K[r1] = self.GenerateKernelBasisElementColumn(theta, i, j, d)
        return K
        
    def GenerateKernelBasisElementColumn(self, theta, i, j, matrix_dim=2):
        # d = matrix_dim
        # 0 <= i,j <= d-1
        # C = kernelcolumn: [theta, d, d]
        # C[0,:,:] = 0
        # C[0,i,j] = 1
        # for k in range(1,theta):
        #   C[k] = RotMat(k*2*pi/theta) * C[0] * RotMat(-k*2*pi/theta) 
        # # K[g v] = g K[v] g^{-1}
        d = matrix_dim
        k = self.k
        
        C = torch.zeros(theta, d, k, requires_grad=False)
        C[0,i,j] = 1
        # TODO: rho 1 -> rho n
        for ind in range(1, theta):
            theta_i = torch.tensor(ind*2*math.pi/theta)
            C[ind] = self.Rho1RotMat(theta_i).matmul(C[0]).matmul(self.RegRotMat(-theta_i.numpy(), k))
        return C
    
    def GenerateKernelBullseyeElement(self, r, theta, matrix_dim=2):
        """
        output: K: [r, theta, d, k]
        """
        d = matrix_dim
        k = self.k
        
        K = torch.zeros(r, theta, d, k, requires_grad=False)
        K[0] = self.GenerateKernelBullseyeElementColumn(theta, d)
        return K
    
    def GenerateKernelBullseyeElementColumn(self, theta, matrix_dim=2):
        d = matrix_dim
        k = self.k
        
        C = torch.zeros(theta, d, k, requires_grad=False)
        C[:] = self.RegToRho1Mat
        return C
            
        
class EquiCtsConv2dRho1ToReg(EquiCtsConvBase):
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, k, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(EquiCtsConv2dRho1ToReg, self).__init__()
        self.num_theta = num_theta
        self.num_radii = num_radii
        self.k = k

        Rho1ToRegMat = EquiLinearRho1ToReg(k).Rho1ToReg    #[k,2]
        self.register_buffer('Rho1ToRegMat', Rho1ToRegMat)
    
        kernel_basis_outer, kernel_bullseye = self.GenerateKernelBasis(num_radii, num_theta, matrix_dim)
        self.register_buffer('kernel_basis_outer', kernel_basis_outer)
        self.register_buffer('kernel_bullseye', kernel_bullseye)
        
        outer_weights = torch.rand(in_channels, out_channels, num_radii, k, matrix_dim)
        outer_weights -= 0.5
        scale_norm = 1 / torch.sqrt(torch.tensor(in_channels, dtype=torch.float))
        outer_weights *= 1 * scale_norm
        self.outer_weights = torch.nn.parameter.Parameter(outer_weights)
        
        bullseye_weights = torch.rand(in_channels, out_channels)
        bullseye_weights -= 0.5
        bullseye_weights *= 1 * scale_norm
        self.bullseye_weights = torch.nn.parameter.Parameter(bullseye_weights)
        
        self.radius = radius

        self.use_attention = use_attention
        self.normalize_attention = normalize_attention
        
    def computeKernel(self):
        # print("[r, d, d, r, theta, k, d] ",self.kernel_basis.shape)
        # print("[c_in,c_out, r, k, d]", self.weights.shape)
        kernel = (torch.einsum('pabrtij,xypab->yxrtij',self.kernel_basis_outer, self.outer_weights) +
                 torch.einsum('rtij,xy->yxrtij',self.kernel_bullseye,self.bullseye_weights))
        return kernel
    
    def GenerateKernelBasis(self, r, theta, matrix_dim=2):
        """
        output: KB : [r+1, k, d, r+1, theta, k, d]  
        KB_bullseye : 
        """ 
        d = matrix_dim
        k = self.k
        
        KB_outer = torch.zeros(r, k, d, r+1, theta, k, d, requires_grad=False)
        K_bullseye = self.GenerateKernelBullseyeElement(r+1, theta, d)
        
        for i in range(k):
            for j in range(d):
                for r1 in range(0, r):
                    KB_outer[r1,i,j] = self.GenerateKernelBasisElement(r+1, theta, i, j, r1+1, d) 
            
        return KB_outer, K_bullseye
    
    def GenerateKernelBasisElement(self, r, theta, i, j, r1, matrix_dim=2):
        """
        output: K: [r, theta, d, k]
        """
        d = matrix_dim
        k = self.k
        
        K = torch.zeros(r, theta, k, d, requires_grad=False)
        K[r1] = self.GenerateKernelBasisElementColumn(theta, i, j, d)
        return K
        
    def GenerateKernelBasisElementColumn(self, theta, i, j, matrix_dim=2):
        # d = matrix_dim
        # 0 <= i,j <= d-1
        # C = kernelcolumn: [theta, d, d]
        # C[0,:,:] = 0
        # C[0,i,j] = 1
        # for k in range(1,theta):
        #   C[k] = RotMat(k*2*pi/theta) * C[0] * RotMat(-k*2*pi/theta) 
        # # K[g v] = g K[v] g^{-1}
        d = matrix_dim
        k = self.k
        
        C = torch.zeros(theta, k, d, requires_grad=False)
        C[0,i,j] = 1
        # TODO: rho 1 -> rho n
        for ind in range(1, theta):
            theta_i = torch.tensor(ind*2*math.pi/theta)
            C[ind] = self.RegRotMat(theta_i.numpy(), k).matmul(C[0]).matmul(self.Rho1RotMat(-theta_i))
        return C
    
    def GenerateKernelBullseyeElement(self, r, theta, matrix_dim=2):
        """
        output: K: [r, theta, d, k]
        """
        d = matrix_dim
        k = self.k
        
        K = torch.zeros(r, theta, k, d, requires_grad=False)
        K[0] = self.GenerateKernelBullseyeElementColumn(theta, d)
        return K
    
    def GenerateKernelBullseyeElementColumn(self, theta, matrix_dim=2):
        d = matrix_dim
        k = self.k
        
        C = torch.zeros(theta, k, d, requires_grad=False)
        C[:] = self.Rho1ToRegMat
        return C
            
        
class EquiCtsConv2dRegToReg(EquiCtsConvBase):
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, k, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(EquiCtsConv2dRegToReg, self).__init__()
        self.num_theta = num_theta
        self.num_radii = num_radii
        self.k = k
    
        
        kernel_basis_outer, kernel_bullseye = self.GenerateKernelBasis(num_radii, num_theta, matrix_dim)
        self.register_buffer('kernel_basis_outer', kernel_basis_outer)
        self.register_buffer('kernel_bullseye', kernel_bullseye)
        
        outer_weights = torch.rand(in_channels, out_channels, num_radii, k, k)
        outer_weights -= 0.5
        scale_norm = 1 / torch.sqrt(torch.tensor(in_channels, dtype=torch.float))
        outer_weights *= 1 * scale_norm
        self.outer_weights = torch.nn.parameter.Parameter(outer_weights)
        
        
        
        bullseye_weights = torch.rand(in_channels, out_channels, k)
        bullseye_weights -= 0.5
        bullseye_weights *= 1 * scale_norm
        self.bullseye_weights = torch.nn.parameter.Parameter(bullseye_weights)
        
        self.radius = radius

        self.use_attention = use_attention
        self.normalize_attention = normalize_attention
        
    def computeKernel(self):
        # print("[r, d, d, r, theta, d, k] ",self.kernel_basis.shape)
        # print("[c_in,c_out, r, d, k]", self.weights.shape)
        kernel = (torch.einsum('pabrtij,xypab->yxrtij',self.kernel_basis_outer, self.outer_weights) +
                 torch.einsum('lrtij,xyl->yxrtij',self.kernel_bullseye,self.bullseye_weights))
        return kernel
    
    def GenerateKernelBasis(self, r, theta, matrix_dim=2):
        """
        output: KB : [r+1, k, k, r+1, theta, k, k]  
        KB_bullseye : 
        """ 
        d = matrix_dim
        k = self.k
        
        KB_outer = torch.zeros(r, k, k, r+1, theta, k, k, requires_grad=False)
        K_bullseye = self.GenerateKernelBullseyeBasis(r+1, theta, d)
        
        for i in range(k):
            for j in range(k):
                for r1 in range(0, r):
                    KB_outer[r1,i,j] = self.GenerateKernelBasisElement(r+1, theta, i, j, r1+1, d) 
            
        return KB_outer, K_bullseye
    
    def GenerateKernelBasisElement(self, r, theta, i, j, r1, matrix_dim=2):
        """
        output: K: [r, theta, k, k]
        """
        d = matrix_dim
        k = self.k
        
        K = torch.zeros(r, theta, k, k, requires_grad=False)
        K[r1] = self.GenerateKernelBasisElementColumn(theta, i, j, d)
        return K
        
    def GenerateKernelBasisElementColumn(self, theta, i, j, matrix_dim=2):
        # d = matrix_dim
        # 0 <= i,j <= d-1
        # C = kernelcolumn: [theta, d, d]
        # C[0,:,:] = 0
        # C[0,i,j] = 1
        # for k in range(1,theta):
        #   C[k] = RotMat(k*2*pi/theta) * C[0] * RotMat(-k*2*pi/theta) 
        # # K[g v] = g K[v] g^{-1}
        d = matrix_dim
        k = self.k
        
        C = torch.zeros(theta, k, k, requires_grad=False)
        C[0,i,j] = 1
        # TODO: rho 1 -> rho n
        for ind in range(1, theta):
            theta_i = torch.tensor(ind*2*math.pi/theta)
            C[ind] = self.RegRotMat(theta_i.numpy(), k).matmul(C[0]).matmul(self.RegRotMat(-theta_i.numpy(), k))
        return C
    
    def GenerateKernelBullseyeBasis(self, r, theta, matrix_dim=2):
        """
        output: K: [k, r, theta, k, k]
        """
        d = matrix_dim
        k = self.k
        
        K = torch.zeros(k, r, theta, k, k, requires_grad=False)
        for l in range(k):
            K[l,0] = self.GenerateKernelBullseyeElementColumn(theta, l, d)
        return K
    
    def GenerateKernelBullseyeElementColumn(self, theta, l, matrix_dim=2):
        d = matrix_dim
        k = self.k
        
        first_col = torch.zeros(k)
        first_col[l] = 1.
        C = torch.zeros(theta, k, k, requires_grad=False)
        C[:] = torch.stack([torch.roll(first_col, i, 0) for i in range(0,self.k)],-1)
        return C
            