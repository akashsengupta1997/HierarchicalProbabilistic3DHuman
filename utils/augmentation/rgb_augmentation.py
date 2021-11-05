import numpy as np
import torch
from torchgeometry.image.gaussian import gaussian_blur


def random_occlude_bottom_half(rgb,
                               joints2D,
                               joints2D_visib,
                               occlude_probability=0.05):
    batch_size = rgb.shape[0]
    wh = rgb.shape[-1]

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            occlude_from = int(wh / 2.0) + np.random.randint(low=-int(wh / 5.), high=int(wh / 5.))
            rgb[i, :, occlude_from:, :] = 0

            if joints2D is not None:
                joints2D_to_occlude = joints2D[i, :, 1] > occlude_from
                joints2D_visib[i, joints2D_to_occlude] = False

    return rgb, joints2D, joints2D_visib


def random_occlude_top_half(rgb,
                            joints2D,
                            joints2D_visib,
                            occlude_probability=0.05):
    batch_size = rgb.shape[0]
    wh = rgb.shape[-1]

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            occlude_up_to = int(wh / 2.0) + np.random.randint(low=-int(wh / 5.), high=int(wh / 5.))
            rgb[i, :, :occlude_up_to, :] = 0

            if joints2D is not None:
                joints2D_to_occlude = joints2D[i, :, 1] < occlude_up_to
                joints2D_visib[i, joints2D_to_occlude] = False

    return rgb, joints2D, joints2D_visib


def random_occlude_vertical_half(rgb,
                                 joints2D,
                                 joints2D_visib,
                                 occlude_probability=0.05):
    batch_size = rgb.shape[0]
    wh = rgb.shape[-1]

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            occlude_up_to = int(wh / 2.0) + np.random.randint(low=-int(wh / 30.), high=int(wh / 30.))
            if np.random.rand() > 0.5:
                rgb[i, :, :, :occlude_up_to] = 0
                if joints2D is not None:
                    joints2D_to_occlude = joints2D[i, :, 0] < occlude_up_to
            else:
                rgb[i, :, :, occlude_up_to:] = 0
                if joints2D is not None:
                    joints2D_to_occlude = joints2D[i, :, 0] > occlude_up_to
            if joints2D is not None:
                joints2D_visib[i, joints2D_to_occlude] = False

    return rgb, joints2D, joints2D_visib


def random_pixel_noise_per_channel(rgb,
                                   per_channel_pixel_noise_factor=0.2):
    l, h = 1 - per_channel_pixel_noise_factor, 1 + per_channel_pixel_noise_factor
    pixel_noise = (h - l) * torch.rand(rgb.shape[0], 3, device=rgb.device) + l
    rgb = torch.clamp(rgb * pixel_noise[:, :, None, None], max=1.0)

    return rgb


def random_gaussian_blur(rgb,
                         sigma_range=(0.2, 1.2),
                         kernel_size=7):
    l, h = sigma_range
    sigma = (h - l) * np.random.rand() + l
    # Note this is currently applying same gaussian kernel to all images in the batch.
    rgb = gaussian_blur(src=rgb,
                        sigma=(sigma, sigma),
                        kernel_size=(kernel_size, kernel_size))
    return rgb


def augment_rgb(rgb,
                joints2D,
                joints2D_visib,
                rgb_augment_config):

    # Occlude bottom/top/left halves of the image (body AND background)
    rgb, joints2D, joints2D_visib = random_occlude_bottom_half(rgb=rgb,
                                                               joints2D=joints2D,
                                                               joints2D_visib=joints2D_visib,
                                                               occlude_probability=rgb_augment_config.OCCLUDE_BOTTOM_PROB)
    rgb, joints2D, joints2D_visib = random_occlude_top_half(rgb=rgb,
                                                            joints2D=joints2D,
                                                            joints2D_visib=joints2D_visib,
                                                            occlude_probability=rgb_augment_config.OCCLUDE_TOP_PROB)
    rgb, joints2D, joints2D_visib = random_occlude_vertical_half(rgb=rgb,
                                                                 joints2D=joints2D,
                                                                 joints2D_visib=joints2D_visib,
                                                                 occlude_probability=rgb_augment_config.OCCLUDE_VERTICAL_PROB)

    # Per channel (RGB) pixel noise
    rgb = random_pixel_noise_per_channel(rgb=rgb,
                                         per_channel_pixel_noise_factor=rgb_augment_config.PIXEL_CHANNEL_NOISE)

    return rgb, joints2D, joints2D_visib
