import train_utils as tu
from curriculum import CelebAHQ_min, extract_metadata

import math
import torch
from tqdm import tqdm
from siren import SIREN
from loss_layers import SSIM
from generator import Generator
from torch_ema import ExponentialMovingAverage
from discriminator import CCSEncoderDiscriminator


def train(rank):
    CHANNELS = 3
    N_EPOCHS = 3000
    LATENT_DIM = 256

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device(rank)
    metadata = extract_metadata(CelebAHQ_min, 0)


    z = tu.z_sampler((20, 256), device='cpu')
    
    generator = Generator(SIREN, z_dim=LATENT_DIM, use_aux=True).to(device)
    discriminator = CCSEncoderDiscriminator().to(device)

    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    ssim = SSIM().to(device)

    # use generator_ddp 
    optimizer_G = torch.optim.Adam(generator.parameters(), 
                                   lr=metadata['gen_lr'], 
                                   betas=metadata['betas'], 
                                   weight_decay=metadata['weight_decay'])
    
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                    lr=metadata['disc_lr'], 
                                    betas=metadata['betas'],
                                    weight_decay=metadata['weight_decay'])
    
    losses_G = []
    losses_D = []

    generator.set_device(device)

    torch.manual_seed(rank)
    total_progress_bar = tqdm(total=N_EPOCHS, 
                              desc='Total progress', 
                              dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    for epoch in range(N_EPOCHS):
        total_progress_bar.update(1)
        metadata = extract_metadata(CelebAHQ_min, discriminator.step)

        tu.set_generator_opt_params(optimizer_G, metadata)
        tu.set_discriminator_opt_params(optimizer_D, metadata)

        step_last_upsample = 0
        for i in range(5):
            BS = metadata['batch_size']
            W_H = metadata['img_size']

            imgs = torch.rand(BS, 
                            CHANNELS,
                            W_H,
                            W_H)
            
            metadata = extract_metadata(CelebAHQ_min, discriminator.step)

            # if dataloader.batch_size != metadata['batch_size']: break

            generator.train()
            discriminator.train()
            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))
            
            real_imgs = imgs.to(device, non_blocking=True)
            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = tu.z_sampler((BS, LATENT_DIM), device=device)
                    split_bs = BS // metadata['batch_split']

                    gen_imgs = []
                    gen_positions = []

                    for split in range(metadata['batch_split']):
                        sub_z = z[split * split_bs:(split+1) * split_bs]
                        g_imgs, g_pos, _, _ = generator(sub_z, alpha=alpha, **metadata)


                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                assert real_imgs.shape == gen_imgs.shape

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator(real_imgs, alpha, **metadata)

            
            grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
            inv_scale = 1./scaler.get_scale()
            grad_real = [p * inv_scale for p in grad_real][0]
            

            with torch.cuda.amp.autocast():
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty

                g_preds, g_pred_latent, g_pred_position = discriminator(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty=0

                loss_D = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                losses_D.append(losses_D.item())


            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.unscale_(optimizer_D)          
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), metadata['grad_clip'])
            scaler.step()
            print('Discriminator Loss:', losses_D)

            # TRAIN GENERATOR

            z = tu.z_sampler((BS, LATENT_DIM), device=device)
            split_bs = BS // metadata['batch_split']
            for split in range(split_bs):
                with torch.cuda.amp.autocast():
                    sub_z = z[split * sub_z:(split+1) * sub_z]
                    gen_imgs, gen_positions, gen_init_imgs, gen_warp_imgs= generator(sub_z, alpha=alpha, **metadata)
                    g_preds, g_pred_latent, g_pred_position = discriminator(gen_imgs, alpha, **metadata)
                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])
                    
                    g_preds = torch.topk(g_preds, topk_num, dim=0).values
                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty=0

                    # reproj lambda
                    pred = (gen_warp_imgs + 1) / 2
                    target = (gen_init_imgs + 1) / 2
                    abs_diff = torch.abs(target - pred)
                    l1_loss = abs_diff.mean(1, True)

                    ssim_loss = ssim(pred, target).mean(1, True)
                    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                    reprojection_loss = reprojection_loss.mean() * metadata['reproj_lambda']

                    loss_G = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty + reprojection_loss
                    losses_G.append(loss_G.item())
                    print('Generator Loss:', losses_G)
            
                scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()

            ema.update(generator.parameters())
            ema2.update(generator.parameters())
        
        break

if __name__ == '__main__':
    train(0)