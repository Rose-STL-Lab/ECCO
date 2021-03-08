import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from EquiCtsConv import *


class RelEquiCtsConv2d(EquiCtsConv2d):
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(RelEquiCtsConv2d, self).__init__(in_channels, out_channels, radius, num_radii, num_theta, 
                                               matrix_dim, use_attention, normalize_attention)
        
    def ContinuousConv(
        self, field, center, field_feat, 
        field_mask, ctr_feat
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
        
        field_feat = field_feat.unsqueeze(1) - ctr_feat.unsqueeze(2)
        attention_field_feat = field_feat * attention.unsqueeze(-1)
        # attention_field_feat: [batch, num_m, num_n, c_in, 2]

        # print(kernel_on_field.shape, attention_field_feat.shape)
        out = torch.einsum('bmnoiyx,bmnix->bmoy', kernel_on_field, attention_field_feat)
        # out: [batch, num_m, c_out, 2]
        
        return out / psi
    
    def forward(
        self, field, center, field_feat, 
        field_mask, ctr_feat, normalize_attention=False
    ):
        out = self.ContinuousConv(
            field, center, field_feat, field_mask, 
            ctr_feat
        )
        return out

class RelEquiCtsConv2dRegToRho1(EquiCtsConv2dRegToRho1):
    
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, k, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(RelEquiCtsConv2dRegToRho1, self).__init__(in_channels, out_channels, radius, num_radii, num_theta, 
                                                        k, matrix_dim, use_attention, normalize_attention)
    
    def ContinuousConv(
        self, field, center, field_feat, 
        field_mask, ctr_feat
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
        
        field_feat = field_feat.unsqueeze(1) - ctr_feat.unsqueeze(2)
        
        attention_field_feat = field_feat*attention.unsqueeze(-1)
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
        
class RelEquiCtsConv2dRho1ToReg(EquiCtsConv2dRho1ToReg):
    
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, k, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(RelEquiCtsConv2dRho1ToReg, self).__init__(in_channels, out_channels, radius, num_radii, num_theta, 
                                                        k, matrix_dim, use_attention, normalize_attentionv)
        
       
    def ContinuousConv(
        self, field, center, field_feat, 
        field_mask, ctr_feat
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
        
        field_feat = field_feat.unsqueeze(1) - ctr_feat.unsqueeze(2)
        
        attention_field_feat = field_feat*attention.unsqueeze(-1)
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

class RelEquiCtsConv2dRegToReg(EquiCtsConv2dRegToReg):
    
    def __init__(self, in_channels, out_channels, radius, num_radii, num_theta, k, matrix_dim=2, 
                 use_attention=True, normalize_attention=True):
        super(RelEquiCtsConv2dRegToReg, self).__init__(in_channels, out_channels, radius, num_radii, num_theta, 
                                                        k, matrix_dim, use_attention, normalize_attention)
        self.kernel = self.computeKernel()
        
    
    def ContinuousConv(
        self, field, center, field_feat, 
        field_mask, ctr_feat
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
        
        field_feat = field_feat.unsqueeze(1) - ctr_feat.unsqueeze(2)
        
        attention_field_feat = field_feat*attention.unsqueeze(-1)
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
    