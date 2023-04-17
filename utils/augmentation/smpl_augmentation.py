import torch

from smplx.lbs import batch_rodrigues


def uniform_sample_shape(batch_size, mean_shape, delta_betas_range):
    """
    Uniform sampling of shape parameter deviations from the mean.
    """
    l, h = delta_betas_range
    delta_betas = (h-l)*torch.rand(batch_size, mean_shape.shape[0], device= mean_shape.device) + l
    shape = delta_betas + mean_shape
    return shape  # (bs, num_smpl_betas)


def normal_sample_shape(batch_size, mean_shape, std_vector):
    """
    Gaussian sampling of shape parameter deviations from the mean.
    """
    shape = mean_shape + torch.randn(batch_size, mean_shape.shape[0], device=mean_shape.device)*std_vector
    return shape  # (bs, num_smpl_betas)


def uniform_random_unit_vector(num_vectors):
    """
    Uniform sampling random 3D unit-vectors, i.e. points on unit sphere.
    """
    e = torch.randn(num_vectors, 3)
    e = torch.div(e, torch.norm(e, dim=-1, keepdim=True))
    return e  # (num_vectors, 3)

