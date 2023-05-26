import os
import shutil
import torch
import copy
import argparse

from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm

from train_utils import get_dataset

def output_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1

def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1

def setup_evaluation(dataset_name, generated_dir, save_dir, dataset_path, target_size=128, num_imgs=512):
    real_dir = os.path.join(save_dir, 'EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        
    # dataloader, CHANNELS = datasets.get_dataset(dataset_name, output_size=target_size, dataset_path=dataset_path)
    dataloader = get_dataset(dataset_path, evaluate=True, metadata={'output_size': target_size, 'batch_size': 32})
    print('outputting real images...')
    output_real_images(dataloader, num_imgs, real_dir)
    print('...done')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir
    
def output_images(alpha, generator, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    # metadata['img_size'] = 128
    metadata['batch_size'] = 2

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 0.6

    img_counter = rank
    generator.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z = torch.randn((metadata['batch_size'], generator.z_dim), device=generator.device)
            generated_imgs, _, _, _ = generator(z, alpha=alpha, **metadata)

            for img in generated_imgs:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()
    
def calculate_fid(dataset_name, generated_dir, real_dir, target_size=256):
    # real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    # fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
    real_dir = 'eval/EvalImages/CelebAHQ_real_images_64'
    generated_dir = 'outputs/evaluation/generated'
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
    torch.cuda.empty_cache
    return fid