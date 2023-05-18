import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, frequency, phase):
        x = self.proj(x)
        frequency = frequency.unsqueeze(1).expand_as(x)
        phase = phase.unsqueeze(1).expand_as(x)

        # adjust freq
        x = frequency * x + phase
        return torch.sin(x)
    
class MappingNet(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, negative_slope=0.2, init_weight_clip=0.25) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope, inplace=True)
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

        self.layers.apply(self.init_kaiming)
        self.proj.apply(self.init_kaiming)

        with torch.no_grad(): self.proj.weight *= init_weight_clip

    def init_kaiming(self, m):
        if m.__class__.__name__.find('Linear') != -1:
            torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, z):
        codes = self.layers(z)
        offsets = self.proj(codes)
        cutoff = offsets.shape[-1]//2

        frequencies = offsets[..., :cutoff]
        phases = offsets[..., cutoff:]

        return frequencies, phases, codes
    
class SIREN(nn.Module):
    def __init__(self, input_dim=3, z_dim=256, hidden_dim=256, output_dim=4, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.films = nn.ModuleList([
            FiLM(input_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim),
            FiLM(hidden_dim, hidden_dim)
        ])
        self.proj = nn.Linear(hidden_dim, 1)
         
        self.color_film = FiLM(hidden_dim + 3, hidden_dim) # +3 due to rays
        self.color_proj = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = MappingNet(z_dim, hidden_dim=256, output_dim=(len(self.films) + 1)*hidden_dim*2)

        self.films.apply(self.init_frequency(25))
        self.proj.apply(self.init_frequency(25))
        self.color_film.apply(self.init_frequency(25))
        self.color_proj.apply(self.init_frequency(25))
        self.films[0].apply(self.init_frequency(25, first=True))

    def init_frequency(self, freq, first=False):
        def init(m):
            with torch.no_grad():
                if isinstance(m, nn.Linear):
                    num_input = m.weight.size(-1)
                    if first:
                        m.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
        return init
    

    def forward(self, x, z, rays, omega=15, beta=30):
        '''
            Inputs
            x is a 3D ray samples vector of shape (BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * NUM_RAY_SAMPLES, 3)
            z is latent vector of shape (BATCH_SIZE, LATENT_DIM=256)
            rays is sampled ray directions og shape (BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * NUM_RAY_SAMPLES, 3) 
                Note: I thought ray samples needed to be 2D. Don't know why this is 3D
            
            x and rays are sampled view points and directions. Z is a latent vector.
            SIREN's job is to take camera view points and a latent dimension z and output
                1. Concatenated RBG and sigma (intensity) vectors (BATCH_SIZE, IMG_SIZE * IMG_SIZE, 3 + 1)
                2. RBG features (BATCH_SIZE, IMG_SIZE * IMG_SIZE, HIDDEN_DIM)
                3. Embeddings of the mapping network (BATCH_SIZE, HIDDEN_DIM)

            MappingNet takes z and widens it via series of Linear layers. Its output dimension is:
            (N_FILM_LAYERS + 1) * 2 * hidden_dim. Reason for that is, the mapping network's output will
            be sliced to supply DISTINCT frequency and phase vectors to FiLM layer. Each of these frequency
            and phase vectors have hidden_dim dimensions. So we need N_FILM_LAYERS * hidden_dim * 2 length.
            Lastly, we need a final frequency & phase vectors for last color FiLM layer. So in total
            we need a vector of size (N_FILM_LAYERS + 1) * 2 * hidden_dim
        '''
        freqs, phases, codes = self.mapping_network(z)

        freqs = omega * freqs + beta


        for index, film in enumerate(self.films):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim

            f_slice = freqs[..., start:end]
            p_slice = phases[..., start:end]

            x = film(x, f_slice, p_slice)

        intensity = self.proj(x)
        ray_aug = torch.cat([rays, x], dim=-1)

        color_freqs, color_phases = freqs[..., -self.hidden_dim:], phases[..., -self.hidden_dim:]

        rbg_features = self.color_film(ray_aug, color_freqs, color_phases)
        rbg = self.color_proj(rbg_features)
        return torch.cat([rbg, intensity], dim=-1), rbg_features, codes