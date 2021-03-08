import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argoverse

import sys
import os
sys.path.append(os.path.dirname(__file__))

from EquiCtsConv import *
from EquiLinear import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ECCONetwork(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 radius_scale = 40,
                 timestep = 0.1,
                 encoder_hidden_size = 19, 
                 layer_channels = [16, 32, 32, 32, 1]
                 ):
        super(ECCONetwork, self).__init__()
        
        # init parameters
        
        self.num_radii = num_radii
        self.num_theta = num_theta
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.layer_channels = layer_channels
        
        self.encoder_hidden_size = encoder_hidden_size
        self.in_channel = 1 + self.encoder_hidden_size
        self.activation = F.relu
        # self.relu_shift = torch.nn.parameter.Parameter(torch.tensor(0.2))
        relu_shift = torch.tensor(0.2)
        self.register_buffer('relu_shift', relu_shift)
        
        # create continuous convolution and fully-connected layers
        
        convs = []
        denses = []
        # c_in, c_out, radius, num_radii, num_theta
        self.conv_fluid = EquiCtsConv2d(in_channels = self.in_channel, 
                                        out_channels = self.layer_channels[0],
                                        num_radii = self.num_radii, 
                                        num_theta = self.num_theta,
                                        radius = self.radius_scale)
        
        self.conv_obstacle = EquiCtsConv2d(in_channels = 1, 
                                           out_channels = self.layer_channels[0],
                                           num_radii = self.num_radii, 
                                           num_theta = self.num_theta,
                                           radius = self.radius_scale)
        
        self.dense_fluid = EquiLinear(self.in_channel, self.layer_channels[0])
        
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = 3 * self.layer_channels[0] 
        for i in range(1, len(self.layer_channels)):
            out_ch = self.layer_channels[i]
            dense = EquiLinear(in_ch, out_ch)
            denses.append(dense)
            conv = EquiCtsConv2d(in_channels = in_ch, 
                                 out_channels = out_ch,
                                 num_radii = self.num_radii, 
                                 num_theta = self.num_theta,
                                 radius = self.radius_scale)
            convs.append(conv)
            in_ch = self.layer_channels[i]
        
        self.convs = nn.ModuleList(convs)
        self.denses = nn.ModuleList(denses)
        
            
    def update_pos_vel(self, p0, v0, a):
        """Apply acceleration and integrate position and velocity.
        Assume the particle has constant acceleration during timestep.
        Return particle's position and velocity after 1 unit timestep."""
        
        dt = self.timestep
        v1 = v0 + dt * a
        p1 = p0 + dt * (v0 + v1) / 2
        return p1, v1

    def apply_correction(self, p0, p1, correction):
        """Apply the position correction
        p0, p1: the position of the particle before/after basic integration. """
        dt = self.timestep
        p_corrected = p1 + correction
        v_corrected = (p_corrected - p0) / dt
        return p_corrected, v_corrected

    def compute_correction(self, p, v, other_feats, box, box_feats, fluid_mask, box_mask):
        """Precondition: p and v were updated with accerlation"""

        fluid_feats = [v.unsqueeze(-2)]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, -2)

        # compute the correction by accumulating the output through the network layers
        output_conv_fluid = self.conv_fluid(p, p, fluid_feats, fluid_mask)
        output_dense_fluid = self.dense_fluid(fluid_feats)
        output_conv_obstacle = self.conv_obstacle(box, p, box_feats.unsqueeze(-2), box_mask)
        
        feats = torch.cat((output_conv_obstacle, output_conv_fluid, output_dense_fluid), -2)
        # self.outputs = [feats]
        output = feats
        
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            mags = (torch.sum(output**2,axis=-1) + 1e-6).unsqueeze(-1)
            in_feats = output/mags * self.activation(mags - self.relu_shift)
            # in_feats = self.activation(output)
            # in_feats = output
            output_conv = conv(p, p, in_feats, fluid_mask)
            output_dense = dense(in_feats)
            
            # if last dim size of output from cur dense layer is same as last dim size of output
            # current output should be based off on previous output
            if output_dense.shape[-2] == output.shape[-2]:
                output = output_conv + output_dense + output
            else:
                output = output_conv + output_dense
            # self.outputs.append(output)

        # compute the number of fluid particle neighbors.
        # this info is used in the loss function during training.
        # TODO: test this block of code
        self.num_fluid_neighbors = torch.sum(fluid_mask, dim = -1) - 1
    
        # self.last_features = self.outputs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * output
        return self.pos_correction
    
    def forward(self, inputs, states=None):
        """ inputs: 8 elems tuple
        p0_enc, v0_enc, p0, v0, a, feats, box, box_feats
        v0_enc: [batch, num_part, timestamps, 2]
        Computes 1 simulation timestep"""
        p0_enc, v0_enc, p0, v0, a, other_feats, box, box_feats, fluid_mask, box_mask = inputs
            
        if states is None:
            if other_feats is None:
                feats = v0_enc
            else:
                feats = torch.cat((other_feats, v0_enc), -2)
        else:
            if other_feats is None:
                feats = v0_enc
                feats = torch.cat((states[0][...,1:,:], feats), -2)
            else:
                feats = torch.cat((other_feats, states[0][...,1:,:], v0_enc), -2)
        # print(feats.shape)

        # a = (v0 - v0_enc[...,-1,:]) / self.timestep
        p1, v1 = self.update_pos_vel(p0, v0, a)
        pos_correction = self.compute_correction(p1, v1, feats, box, box_feats, fluid_mask, box_mask)
        p_corrected, v_corrected = self.apply_correction(p0, p1, pos_correction.squeeze(-2))

        return p_corrected, v_corrected, (feats, None)

