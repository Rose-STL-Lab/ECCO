#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import importlib
import torch
import pickle


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_utils import *


def get_agent(pr, gt, pr_id, gt_id, agent_id, device='cpu'):
        
    pr_agent = pr[pr_id == agent_id,:]
    gt_agent = gt[gt_id == agent_id,:]
    
    return pr_agent, gt_agent


def evaluate(model, val_dataset, train_window=3, max_iter=2500, device='cpu', start_iter=0, 
             batch_size=32):
    
    print('evaluating.. ', end='', flush=True)
        
    count = 0
    prediction_gt = {}
    losses = []
    val_iter = iter(val_dataset)
    
    for i, sample in enumerate(val_dataset):
        
        if i >= max_iter:
            break
        
        if i < start_iter:
            continue
        
        pred = []
        gt = []

        if count % 1 == 0:
            print('{}'.format(count + 1), end=' ', flush=True)
        
        count += 1
        
        data = {}
        convert_keys = (['pos' + str(i) for i in range(31)] + 
                        ['vel' + str(i) for i in range(31)] + 
                        ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])

        for k in convert_keys:
            data[k] = torch.tensor(np.stack(sample[k])[...,:2], dtype=torch.float32, device=device)


        for k in ['track_id' + str(i) for i in range(31)] + ['city', 'agent_id', 'scene_idx']:
            data[k] = np.stack(sample[k])
        
        for k in ['car_mask', 'lane_mask']:
            data[k] = torch.tensor(np.stack(sample[k]), dtype=torch.float32, device=device).unsqueeze(-1)
            
        scenes = data['scene_idx'].tolist()
            
        data['agent_id'] = data['agent_id'][:,np.newaxis]
        
        data['car_mask'] = data['car_mask'].squeeze(-1)
        accel = torch.zeros(1, 1, 2).to(device)
        data['accel'] = accel

        lane = data['lane']
        lane_normals = data['lane_norm']
        agent_id = data['agent_id']
        city = data['city']
        
        inputs = ([
            data['pos_2s'], data['vel_2s'], 
            data['pos0'], data['vel0'], 
            data['accel'], None,
            data['lane'], data['lane_norm'], 
            data['car_mask'], data['lane_mask']
        ])

        pr_pos1, pr_vel1, states = model(inputs)
        gt_pos1 = data['pos1']

        l = 0.5 * loss_fn(pr_pos1, gt_pos1, 
                          torch.sum(data['car_mask'], dim = -2) - 1, data['car_mask'].squeeze(-1))

        pr_agent, gt_agent = get_agent(pr_pos1, data['pos1'],
                                       data['track_id0'], 
                                       data['track_id1'], 
                                       agent_id, device)
        pred.append(pr_agent.unsqueeze(1).detach().cpu())
        gt.append(gt_agent.unsqueeze(1).detach().cpu())
        del pr_agent, gt_agent
        clean_cache(device)

        pos0 = data['pos0']
        vel0 = data['vel0']
        for i in range(29):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, data['accel'], None, 
                      data['lane'], data['lane_norm'], data['car_mask'], data['lane_mask'])
            pos0, vel0 = pr_pos1, pr_vel1
            pr_pos1, pr_vel1, states = model(inputs, states)
            clean_cache(device)
            
            if i < train_window - 1:
                gt_pos1 = data['pos'+str(i+2)]
                l += 0.5 * loss_fn(pr_pos1, gt_pos1,
                                   torch.sum(data['car_mask'], dim = -2) - 1, data['car_mask'].squeeze(-1))

            pr_agent, gt_agent = get_agent(pr_pos1, data['pos'+str(i+2)],
                                           data['track_id0'], 
                                           data['track_id'+str(i+2)], 
                                           agent_id, device)

            pred.append(pr_agent.unsqueeze(1).detach().cpu())
            gt.append(gt_agent.unsqueeze(1).detach().cpu())
            
            clean_cache(device)
        
        losses.append(l)

        predict_result = (torch.cat(pred, axis=1), torch.cat(gt, axis=1))
        for idx, scene_id in enumerate(scenes):
            prediction_gt[scene_id] = (predict_result[0][idx], predict_result[1][idx])
    
    total_loss = 128 * torch.sum(torch.stack(losses),axis=0) / max_iter
    
    result = {}
    de = {}
    
    for k, v in prediction_gt.items():
        de[k] = torch.sqrt((v[0][:,0] - v[1][:,0])**2 + 
                        (v[0][:,1] - v[1][:,1])**2)
        
    ade = []
    de1s = []
    de2s = []
    de3s = []
    for k, v in de.items():
        ade.append(np.mean(v.numpy()))
        de1s.append(v.numpy()[10])
        de2s.append(v.numpy()[20])
        de3s.append(v.numpy()[-1])
    
    result['ADE'] = np.mean(ade)
    result['ADE_std'] = np.std(ade)
    result['DE@1s'] = np.mean(de1s)
    result['DE@1s_std'] = np.std(de1s)
    result['DE@2s'] = np.mean(de2s)
    result['DE@2s_std'] = np.std(de2s)
    result['DE@3s'] = np.mean(de3s)
    result['DE@3s_std'] = np.std(de3s)

    print(result)
    print('done')

    return total_loss, prediction_gt





