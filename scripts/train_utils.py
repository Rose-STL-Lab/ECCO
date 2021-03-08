import torch
import gc


def euclidean_distance(a, b, epsilon=1e-9):
    return torch.sqrt(torch.sum((a - b)**2, axis=-1) + epsilon)


def loss_fn(pr_pos, gt_pos, num_fluid_neighbors, car_mask):
    gamma = 0.5
    neighbor_scale = 1 / 40
    importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
    dist = euclidean_distance(pr_pos, gt_pos)**gamma
    mask_dist = dist * car_mask
    batch_losses = torch.mean(importance * mask_dist, axis=-1)
    # print(batch_losses)
    return torch.sum(batch_losses)

def clean_cache(device):
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    if device == torch.device('cpu'):
        # gc.collect()
        pass
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def unsqueeze_n(tensor, n):
    for i in range(n):
        tensor = tensor.unsqueeze(-1)
    return tensor
        
    
def normalize_input(tensor_dict, scale, train_window):
    pos_keys = (['pos' + str(i) for i in range(train_window + 1)] + 
                ['pos_2s', 'lane', 'lane_norm'])
    vel_keys = (['vel' + str(i) for i in range(train_window + 1)] + 
                ['vel_2s'])
    max_pos = torch.cat([torch.max(tensor_dict[pk].reshape(tensor_dict[pk].shape[0], -1), axis=-1)
                         .values.unsqueeze(-1) 
                         for pk in pos_keys], axis=-1)
    max_pos = torch.max(max_pos, axis=-1).values
    
    for pk in pos_keys:
        tensor_dict[pk][...,:2] = (tensor_dict[pk][...,:2] - unsqueeze_n(max_pos, len(tensor_dict[pk].shape)-1)) / scale
    
    for vk in vel_keys:
        tensor_dict[vk] = tensor_dict[vk] / scale
        
    return tensor_dict, max_pos

    