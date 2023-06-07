import train_utils as tu
from curriculum import CelebAHQ_min, extract_metadata

import os
import math
import torch
import fid_evaluation
from tqdm import tqdm
from siren import SIREN
from loss_layers import SSIM
from generator import Generator
from torch_ema import ExponentialMovingAverage
from discriminator import CCSEncoderDiscriminator

from datetime import datetime


def train(rank, iteration=None, device='cuda'):
    CHANNELS = 3
    N_EPOCHS = 3000
    LATENT_DIM = 256
    MODEL_SAVE_INTERVAL = 2000
    SAMPLE_INTERVAL = 200
    EVAL_FREQ = 2000
    OUTPUT_DIR = 'outputs'
    EVAL_DIR = 'eval'

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device(device)
    metadata = extract_metadata(CelebAHQ_min, 0 if iteration == None else iteration)


    fixed_z = tu.z_sampler((20, 256), device='cpu')
    ssim = SSIM().to(device)
    


    if iteration == None:
        generator = Generator(SIREN, z_dim=LATENT_DIM, use_aux=True).to(device)
        discriminator = CCSEncoderDiscriminator().to(device)

        
    else:
        print('Recovering model states.')
        generator = torch.load(os.path.join(OUTPUT_DIR, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(OUTPUT_DIR, 'discriminator.pth'), map_location=device)
        
    
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
    
    if iteration != None:
        ema.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'ema.pth'), map_location=device))
        ema2.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'ema2.pth'), map_location=device))
    
    optimizer_G = torch.optim.Adam(generator.parameters(), 
                                   lr=metadata['gen_lr'], 
                                   betas=metadata['betas'], 
                                   weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                    lr=metadata['disc_lr'], 
                                    betas=metadata['betas'],
                                    weight_decay=metadata['weight_decay'])
    
    if iteration != None:
        print('Recovering optimizer states.')
        optimizer_G.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'optimizer_G.pth')))
        optimizer_D.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'optimizer_D.pth')))
        scaler.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'scaler.pth')))
    
        
    losses_G = []
    losses_D = []

    generator.set_device(device)

    torch.manual_seed(rank)
    total_progress_bar = tqdm(total=N_EPOCHS, 
                              desc='Total progress', 
                              dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    
    dataloader = None
    for epoch in range(N_EPOCHS):
        total_progress_bar.update(1)
        metadata = extract_metadata(CelebAHQ_min, discriminator.step)

        tu.set_generator_opt_params(optimizer_G, metadata)
        tu.set_discriminator_opt_params(optimizer_D, metadata)

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, step_next_upsample, step_last_upsample = tu.get_dataset(
                'CelebAMask-HQ/CelebA-HQ-img/*.jpg',
                CelebAHQ_min,
                metadata,
                discriminator.step
            )

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        # sampler.set_epoch(epoch+1)
        for i, (imgs, _) in enumerate(dataloader):
            if discriminator.step > 0 and discriminator.step % MODEL_SAVE_INTERVAL == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(ema.state_dict(), os.path.join(OUTPUT_DIR, '{}_ema.pth'.format(discriminator.step)))
                torch.save(ema2.state_dict(), os.path.join(OUTPUT_DIR, '{}_ema2.pth'.format(discriminator.step)))
                torch.save(generator, os.path.join(OUTPUT_DIR, '{}_generator.pth'.format(discriminator.step)))
                torch.save(discriminator, os.path.join(OUTPUT_DIR, '{}_discriminator.pth'.format(discriminator.step)))
                torch.save(optimizer_G.state_dict(), os.path.join(OUTPUT_DIR, '{}_optimizer_G.pth'.format(discriminator.step)))
                torch.save(optimizer_D.state_dict(), os.path.join(OUTPUT_DIR, '{}_optimizer_D.pth'.format(discriminator.step)))
                torch.save(scaler.state_dict(), os.path.join(OUTPUT_DIR, '{}_scaler.pth'.format(discriminator.step)))
            BS = metadata['batch_size']
            W_H = metadata['img_size']

            metadata = extract_metadata(CelebAHQ_min, discriminator.step)
            if dataloader.batch_size != metadata['batch_size']: 
                print('have batch inconsistency. breaking.')
                break
            if scaler.get_scale() < 1:
                scaler.update(1.)
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

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)
                    
                
                assert real_imgs.shape == gen_imgs.shape, f'Real: {real_imgs.size()}, Gen: {gen_imgs.size()}'
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
                losses_D.append(loss_D.item())


            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.unscale_(optimizer_D)          
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)

            # TRAIN GENERATOR

            z = tu.z_sampler((BS, LATENT_DIM), device=device)
            split_bs = BS // metadata['batch_split']
            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    sub_z = z[split * split_bs:(split+1) * split_bs]
                    gen_imgs, gen_positions, gen_init_imgs, gen_warp_imgs= generator(sub_z, alpha=alpha, **metadata)
                    g_preds, g_pred_latent, g_pred_position = discriminator(gen_imgs, alpha, **metadata)
                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])
                    
                    g_preds = torch.topk(g_preds, topk_num, dim=0).values
                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, sub_z) * metadata['z_lambda']
                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty=0

                    # reproj lambda
                    pred = (gen_warp_imgs + 1) / 2
                    target = (gen_init_imgs + 1) / 2
                    abs_diff = torch.abs(target - pred)
                    l1_loss = abs_diff.mean(1, True)
 
                    ss = ssim(pred, target)
                    ssim_loss = ss.mean(1, True)
                    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                    reprojection_loss = reprojection_loss.mean() * metadata['reproj_lambda']

                    loss_G = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty + reprojection_loss
                    losses_G.append(loss_G.item())
            
                scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()

            ema.update(generator.parameters())
            ema2.update(generator.parameters())
        
            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {OUTPUT_DIR}] [Epoch: {discriminator.epoch}/{N_EPOCHS}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['output_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")

                if discriminator.step % SAMPLE_INTERVAL == 0:
                    
                    generator.eval()
                    tu.ablation(generator, fixed_z, device, alpha, discriminator.step, OUTPUT_DIR, metadata, 'fixed', 0)
                    tu.ablation(generator, fixed_z, device, alpha, discriminator.step, OUTPUT_DIR, metadata, 'tilted', 0, h_mean=0.5)

                    ema.store(generator.parameters())
                    ema.copy_to(generator.parameters())
                    generator.eval()
                    tu.ablation(generator, fixed_z, device, alpha, discriminator.step, OUTPUT_DIR, metadata, 'fixed_ema', 0)
                    tu.ablation(generator, fixed_z, device, alpha, discriminator.step, OUTPUT_DIR, metadata, 'tilted_ema', 0, h_mean=0.5)

                    tu.ablation(generator, fixed_z, device, alpha, discriminator.step, OUTPUT_DIR, metadata, 'random', 0, psi=0.7, random=True)
                    ema.restore(generator.parameters())


                if discriminator.step % SAMPLE_INTERVAL == 0:
                    torch.save(ema.state_dict(), os.path.join(OUTPUT_DIR, 'ema.pth'))
                    torch.save(ema2.state_dict(), os.path.join(OUTPUT_DIR, 'ema2.pth'))
                    torch.save(generator, os.path.join(OUTPUT_DIR, 'generator.pth'))
                    torch.save(discriminator, os.path.join(OUTPUT_DIR, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(OUTPUT_DIR, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(OUTPUT_DIR, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(OUTPUT_DIR, 'scaler.pth'))
                    torch.save(losses_G, os.path.join(OUTPUT_DIR, 'generator.losses'))
                    torch.save(losses_D, os.path.join(OUTPUT_DIR, 'discriminator.losses'))

            if EVAL_FREQ > 0 and (discriminator.step + 1) % EVAL_FREQ == 0:
                generated_dir = os.path.join(OUTPUT_DIR, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation('CelebAHQ', generated_dir, EVAL_DIR, dataset_path='CelebAMask-HQ/CelebA-HQ-img/*.jpg', target_size=metadata['output_size'])
                    
                ema.store(generator.parameters())
                ema.copy_to(generator.parameters())
                generator.eval()
                fid_evaluation.output_images(alpha, generator, metadata, rank, 1, generated_dir)
                ema.restore(generator.parameters())
                if rank == 0:
                    fid = fid_evaluation.calculate_fid('CelebAHQ', generated_dir, EVAL_DIR, target_size=metadata['output_size'])
                    with open(os.path.join(OUTPUT_DIR, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

if __name__ == '__main__':
    train(0)