import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from volumetric_rendering import transform_sampled_points, rgb_feat_integration

class Generator(nn.Module):
    def __init__(self, siren, z_dim, use_aux=True, input_dim=3, output_dim=4) -> None:
        super().__init__()
        self.step = 0
        self.epoch = 0
        self.z_dim = z_dim
        self.siren = siren(input_dim=input_dim, output_dim=output_dim, z_dim=self.z_dim, device=None)
        self.decoder = Decoder()
        self.use_aux = use_aux

    def set_device(self, device):
        self.device = device
        self.siren.device = device
        # self.generate_avg_frequencies()

    def forward(self, z, img_size, output_size, nerf_noise=1, alpha=1, fov=12, ray_start=0.88, ray_end=1.12, num_steps=16, max_mixup_ratio=0.1, **kwargs):
        # alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))
        # nerf noise = max(0, 1. - discriminator.step/5000.)

        batch_size = z.shape[0]

        # 1. Get camera points
        points, depths, origins, origin_norms = self.get_camera(batch_size, img_size, fov, ray_start, ray_end, num_steps)

        # 2. Sample primary viewpoints from camera points
        primary_points, primary_depths, primary_ray_directions, primary_pitch, primary_yaw, primary_cam2world = self.sample_viewpoint(points, depths, origins, batch_size, img_size, num_steps)
        
        # 3. Sample auxiliary viewpoints from camera points
        if self.use_aux:
            aux_points, aux_depths, aux_ray_directions, aux_pitch, aux_yaw, aux_cam2world = self.sample_viewpoint(points, depths, origins, batch_size, img_size, num_steps)

        # 4. Calculate intensity and color for primary using SIREN
        primary_output, primary_rgb_feats, codes = self.siren(primary_points, z, primary_ray_directions)

        # 5. Generate primary image maps
        primary_rgb_feat_maps, primary_initial_rgb, primary_depths = self.generate_image(primary_output, primary_rgb_feats, primary_depths, 
                                                                         batch_size, img_size, num_steps, nerf_noise)

        if self.use_aux:
            # 6. Repeat (4) & (5) for auxiliary
            aux_output, aux_rgb_feats, _ = self.siren(aux_points, z, aux_ray_directions)
            aux_rgb_feat_maps, aux_initial_rgb, _ = self.generate_image(aux_output, aux_rgb_feats, aux_depths, 
                                                                         batch_size, img_size, num_steps, nerf_noise)
            
            # 7. Warp auxiliary to primary to learn consistent camera movement
            warp_rgb_feat_maps, warp_rgb = self.project(origins,
                         primary_depths, 
                         batch_size, 
                         img_size, 
                         primary_cam2world, 
                         aux_cam2world, 
                         origin_norms, 
                         aux_rgb_feat_maps, 
                         aux_initial_rgb)


            # 8. Mixup original and warped images
            sampled_mixup_ratio = torch.rand((batch_size, 1, 1, 1), device=self.device) * max_mixup_ratio
            rgb_feat_maps = (1 - sampled_mixup_ratio) * primary_rgb_feat_maps + sampled_mixup_ratio * warp_rgb_feat_maps
        else:
            # no mixing and no warping
            rgb_feat_maps = primary_rgb_feat_maps
            warp_rgb = None


        # RETURN THESE AFTER IMPLEMENTING 2D DECODER
        
        im = self.decoder(codes, rgb_feat_maps, output_size, alpha)
        return im, torch.cat([primary_pitch, primary_yaw], -1), primary_initial_rgb, warp_rgb

    def get_camera(self, batch_size, img_size, fov, ray_start, ray_end, num_steps):
        ''' creates camera distribution to sample from '''
        with torch.no_grad():
            x_ = torch.linspace(-1, 1, img_size, device=self.device)
            y_ = torch.linspace(-1, 1, img_size, device=self.device)
            x, y = torch.meshgrid(x_, y_)

            x = x.T.flatten()
            y = y.T.flatten()
            z = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)


            origins = torch.stack([x, y, z], dim=-1)
            origin_norms = torch.norm(origins, dim=-1, keepdim=True)
            origins = origins / origin_norms

            depths = torch.linspace(ray_start, ray_end, num_steps, device=self.device)\
                            .reshape(1, num_steps, 1)\
                            .repeat(img_size*img_size, 1, 1)
        
            points = origins.unsqueeze(1).repeat(1, num_steps, 1) * depths

            points = torch.stack(batch_size*[points])
            depths = torch.stack(batch_size*[depths])
            origins = torch.stack(batch_size*[origins]).to(self.device)
        return points, depths, origins, origin_norms
    
    def sample_viewpoint(self, points, depths, origins, batch_size, img_size, num_steps):
        ''' samples a viewpoint from camera distribution '''
        points, depths, ray_directions, ray_origins, pitch, yaw, cam2world = transform_sampled_points(points, depths, origins, 
                                                                                                             h_stddev=0.3, 
                                                                                                             v_stddev=0.155, 
                                                                                                             h_mean=math.pi*0.5, 
                                                                                                             v_mean=math.pi*0.5,
                                                                                                             device=self.device, 
                                                                                                             mode='gaussian')

        ray_directions = torch.unsqueeze(ray_directions, -2)\
                            .expand(-1, -1, num_steps, -1)\
                            .reshape(batch_size, img_size*img_size*num_steps, 3)
        points = points.reshape(batch_size, img_size*img_size*num_steps, 3)

        return points, depths, ray_directions, pitch, yaw, cam2world
    
    def generate_image(self, outputs, rgb_feats, depths, batch_size, img_size, num_steps, nerf_noise):
        ''' generates color and intensity maps from camera ray '''
        outputs = outputs.reshape(batch_size, img_size * img_size, num_steps, 4)
        rgb_feats = rgb_feats.reshape(batch_size, img_size * img_size, num_steps, self.siren.hidden_dim)



        # Create images with NeRF
        initial_rgb, rgb_feat_maps, depth, _ = rgb_feat_integration(outputs, rgb_feats, depths, device=self.device,
                                                                                            white_back=False,
                                                                                            last_back=False, 
                                                                                            clamp_mode='relu', 
                                                                                            noise_std=nerf_noise)
        
        rgb_feat_maps = rgb_feat_maps.reshape(batch_size, img_size, img_size, self.siren.hidden_dim)\
                                            .permute(0, 3, 1, 2)\
                                            .contiguous()

        initial_rgb = initial_rgb.reshape(batch_size, img_size, img_size, 3)\
                                 .permute(0, 3, 1, 2).contiguous()
        
        return rgb_feat_maps, initial_rgb, depth
    
    def project(self, origins, primary_depths, batch_size, img_size, primary_cam2world, aux_cam2world, origin_norms, aux_rgb_feat_maps, aux_initial_rgb):
        ''' warps auxiliary image to primary image '''
        primary_points = origins.reshape((batch_size, img_size, img_size, 3)) * primary_depths.reshape((batch_size, img_size, img_size, 1))
        
        primary_points_opaque = torch.ones((batch_size, img_size, img_size, 4), device=self.device)
        primary_points_opaque[:, :, :, :3] = primary_points

        aux_cam2world_inv = torch.inverse(aux_cam2world.float())
        T = aux_cam2world_inv @ primary_cam2world
        primary_proj_to_aux = torch.bmm(T, primary_points_opaque.reshape(batch_size, -1, 4).permute(0,2,1))\
                                .permute(0, 2, 1)\
                                .reshape(batch_size, img_size, img_size, 4)

        X = primary_proj_to_aux[..., 0:1]
        Y = primary_proj_to_aux[..., 1:2]
        Z = primary_proj_to_aux[..., 2:3]

        primary_grid_in_aux = torch.cat((-X / Z, Y / Z), -1) * origin_norms.reshape(1, img_size, img_size, 1)
        warp_rgb_feat_maps = F.grid_sample(aux_rgb_feat_maps, primary_grid_in_aux, align_corners=True) 
        warp_rgb = F.grid_sample(aux_initial_rgb, primary_grid_in_aux, align_corners=True) 
        return warp_rgb_feat_maps, warp_rgb

