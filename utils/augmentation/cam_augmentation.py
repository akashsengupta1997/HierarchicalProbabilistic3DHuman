import torch


def augment_cam_t(mean_cam_t, xy_std=0.05, delta_z_range=(-0.5, 0.5)):
    batch_size = mean_cam_t.shape[0]
    device = mean_cam_t.device
    new_cam_t = mean_cam_t.clone()
    delta_tx_ty = torch.randn(batch_size, 2, device=device) * xy_std
    new_cam_t[:, :2] = mean_cam_t[:, :2] + delta_tx_ty

    l, h = delta_z_range
    delta_tz = (h - l) * torch.rand(batch_size, device=device) + l
    new_cam_t[:, 2] = mean_cam_t[:, 2] + delta_tz

    return new_cam_t