import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    steps = list(filter(lambda x: type(x) == int, curriculum.keys()))
    for prev, next in zip(steps[:-1], steps[1:]):
        if prev <= current_step < next:
            return next
    return 100000    

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['output_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['output_size'] == current_size:
            return curriculum_step
    return 0

def extract_metadata(curriculum, current_step):
    return_dict = {} 
    # process batch_size, num_steps, first sort list, then for every list...
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break

    # process other keys like dataset_path, fov and so on
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

CelebAHQ = {
    0: {'batch_size': 28 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4, 'pos_lambda': 15, 'reproj_lambda': 1},
    int(25e3): {'batch_size': 26 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 128, 'batch_split': 4, 'gen_lr': 5e-5, 'disc_lr': 2e-4, 'pos_lambda': 15, 'reproj_lambda': 1},
    int(50e3): {'batch_size': 22 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 256, 'batch_split': 4, 'gen_lr': 3e-5, 'disc_lr': 1e-4, 'pos_lambda': 0, 'reproj_lambda': 1},
    int(100e3): {'batch_size': 14 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 512, 'batch_split': 4, 'gen_lr': 2e-5, 'disc_lr': 0.8e-4, 'pos_lambda': 0, 'reproj_lambda': 1},
    int(200e3): {},
    'dataset_path': '/data/xuanmeng/dataset/CelebAMask-HQ/CelebA-HQ-img/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 2000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'dataset': 'CelebAHQ',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'stereo_auxiliary': True,
    'max_mixup_ratio' : 0.1,
    'z_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
}

CelebAHQ_min = {
    0: {'batch_size': 28 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4, 'pos_lambda': 15, 'reproj_lambda': 1},
    int(25e3): {'batch_size': 26 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 128, 'batch_split': 4, 'gen_lr': 5e-5, 'disc_lr': 2e-4, 'pos_lambda': 15, 'reproj_lambda': 1},
    int(50e3): {'batch_size': 22 * 2, 'num_steps': 16, 'img_size': 64, 'output_size': 256, 'batch_split': 4, 'gen_lr': 3e-5, 'disc_lr': 1e-4, 'pos_lambda': 0, 'reproj_lambda': 1},
    'dataset_path': '/data/xuanmeng/dataset/CelebAMask-HQ/CelebA-HQ-img/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 2000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'dataset': 'CelebAHQ',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'stereo_auxiliary': True,
    'max_mixup_ratio' : 0.1,
    'z_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
}


if __name__ == '__main__':
    for step in [0, 24499, 25000, 49999, 50000, 99999, 100000]:
        next_ = next_upsample_step(CelebAHQ_min, step)
        last_ = last_upsample_step(CelebAHQ_min, step)
        print(f'[Step {step}] Next: {next_}, Last: {last_}')