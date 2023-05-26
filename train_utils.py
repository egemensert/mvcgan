import os
import copy
import torch
from torchvision.utils import save_image

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

def ablation(generator, fixed_z, device, alpha, discriminator_step, output_dir, metadata, suffix, h_stddev, h_mean=None, psi=None, fix_row=5, fix_num=20, random=False):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = h_stddev
            if h_mean != None:
                copied_metadata['h_mean'] += h_mean
            
            if psi != None:
                copied_metadata['psi'] = psi
            gen_imgs = []
            arg = None
            if random:
                rand_z = torch.randn_like(fixed_z)
                arg = rand_z
            else:
                arg = fixed_z

            for idx in range(fixed_z.shape[0]):
                g_imgs, _, _, _ = generator(arg[idx:idx+1].to(device), alpha=alpha, **copied_metadata)
                gen_imgs.append(g_imgs)
            
            gen_imgs = torch.cat(gen_imgs, axis=0)
            gen_imgs = ((gen_imgs + 1) / 2).float()
            gen_imgs = gen_imgs.clamp_(0, 1)

    save_image(gen_imgs[:fix_num], os.path.join(output_dir, f"{discriminator_step}_{suffix}.png"), nrow=fix_row, normalize=True)