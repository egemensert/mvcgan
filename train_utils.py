import os
import torch

def z_sampler(shape, device):
    return torch.randn(shape, device=device)

def set_generator_opt_params(optimizer_G, metadata):
    for param_group in optimizer_G.param_groups:
        # reduce mapping networks LR to 5% of rest of network's lr.
        if param_group.get('name', None) == 'mapping_network':
            print('Mapping network found!')
            param_group['lr'] = metadata['gen_lr'] * 5e-2
        else:
            param_group['lr'] = metadata['gen_lr']
        
        param_group['betas'] = metadata['betas']
        param_group['weight_decay'] = metadata['weight_decay']

def set_discriminator_opt_params(optimizer_D, metadata):
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = metadata['disc_lr']
        param_group['betas'] = metadata['betas']
        param_group['weight_decay'] = metadata['weight_decay']