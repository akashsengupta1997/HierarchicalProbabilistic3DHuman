import numpy as np
import torch


COCO_JOINTS = {
    'Right Ankle': 16, 'Right Knee': 14, 'Right Hip': 12,
    'Left Hip': 11, 'Left Knee': 13, 'Left Ankle': 15,
    'Right Wrist': 10, 'Right Elbow': 8, 'Right Shoulder': 6,
    'Left Shoulder': 5, 'Left Elbow': 7, 'Left Wrist': 9,
    'Right Ear': 4, 'Left Ear': 3, 'Right Eye': 2, 'Left Eye': 1,
    'Nose': 0
}

# The SMPL model (im smpl_official.py) returns a large superset of joints.
# Different subsets are used during training - e.g. H36M 3D joints convention and COCO 2D joints convention.
# Joint label conversions from SMPL to H36M/COCO/LSP
ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # Using OP Hips
ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

# Joint label and body part seg label matching
# 24 part seg: COCO Joints
TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP = {19: 7,
                                          21: 7,
                                          20: 8,
                                          22: 8,
                                          4: 9,
                                          3: 10,
                                          12: 13,
                                          14: 13,
                                          11: 14,
                                          13: 14,
                                          5: 15,
                                          6: 16}


def convert_densepose_seg_to_14part_labels(densepose_seg):
    """
    Convert 24 body-part labels (DensePose convention) to 14 body-part labels.
    """
    if isinstance(densepose_seg, torch.Tensor):
        fourteen_part_seg = torch.zeros_like(densepose_seg)
    elif isinstance(densepose_seg, np.ndarray):
        fourteen_part_seg = np.zeros_like(densepose_seg)

    fourteen_part_seg[densepose_seg == 1] = 1
    fourteen_part_seg[densepose_seg == 2] = 1
    fourteen_part_seg[densepose_seg == 3] = 11
    fourteen_part_seg[densepose_seg == 4] = 12
    fourteen_part_seg[densepose_seg == 5] = 14
    fourteen_part_seg[densepose_seg == 6] = 13
    fourteen_part_seg[densepose_seg == 7] = 8
    fourteen_part_seg[densepose_seg == 8] = 6
    fourteen_part_seg[densepose_seg == 9] = 8
    fourteen_part_seg[densepose_seg == 10] = 6
    fourteen_part_seg[densepose_seg == 11] = 9
    fourteen_part_seg[densepose_seg == 12] = 7
    fourteen_part_seg[densepose_seg == 13] = 9
    fourteen_part_seg[densepose_seg == 14] = 7
    fourteen_part_seg[densepose_seg == 15] = 2
    fourteen_part_seg[densepose_seg == 16] = 4
    fourteen_part_seg[densepose_seg == 17] = 2
    fourteen_part_seg[densepose_seg == 18] = 4
    fourteen_part_seg[densepose_seg == 19] = 3
    fourteen_part_seg[densepose_seg == 20] = 5
    fourteen_part_seg[densepose_seg == 21] = 3
    fourteen_part_seg[densepose_seg == 22] = 5
    fourteen_part_seg[densepose_seg == 23] = 10
    fourteen_part_seg[densepose_seg == 24] = 10

    return fourteen_part_seg


def convert_multiclass_to_binary_labels(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    if isinstance(multiclass_labels, torch.Tensor):
        binary_labels = torch.zeros_like(multiclass_labels)
    elif isinstance(multiclass_labels, np.ndarray):
        binary_labels = np.zeros_like(multiclass_labels)

    binary_labels[multiclass_labels != 0] = 1

    return binary_labels


def convert_2Djoints_to_gaussian_heatmaps(joints2D, img_wh, std=4):
    """
    :param joints2D: (N, 2) array, 2D joint locations.
    :return heatmaps: (img_wh, img_wh, N) array, 2D joint heatmaps (channels last).
    """
    xx, yy = np.meshgrid(np.arange(img_wh),
                         np.arange(img_wh))
    xx = xx[None, :, :].astype(np.float32)
    yy = yy[None, :, :].astype(np.float32)

    j2d_u = joints2D[:, 0, None, None]
    j2d_v = joints2D[:, 1, None, None]
    heatmap = np.exp(-(((xx - j2d_u) / std) ** 2) / 2 - (((yy - j2d_v) / std) ** 2) / 2).transpose(1, 2, 0)
    return heatmap


def convert_2Djoints_to_gaussian_heatmaps_torch(joints2D,
                                                img_wh,
                                                std=4):
    """
    :param joints2D: (B, N, 2) tensor - batch of 2D joints.
    :param img_wh: int, dimensions of square heatmaps
    :param std: standard deviation of gaussian blobs
    :return heatmaps: (B, N, img_wh, img_wh) - batch of 2D joint heatmaps (channels first).
    """
    device = joints2D.device

    xx, yy = torch.meshgrid(torch.arange(img_wh, device=device),
                            torch.arange(img_wh, device=device))
    xx = xx[None, None, :, :].float()
    yy = yy[None, None, :, :].float()

    j2d_u = joints2D[:, :, 0, None, None]  # Horizontal coord (columns)
    j2d_v = joints2D[:, :, 1, None, None]  # Vertical coord (rows)
    heatmap = torch.exp(-(((xx - j2d_v) / std) ** 2) / 2 - (((yy - j2d_u) / std) ** 2) / 2)
    return heatmap


def convert_heatmaps_to_2Djoints_coordinates_torch(joints2D_heatmaps,
                                                   eps=1e-6):
    """
    Convert 2D joint heatmaps into coordinates using argmax.
    :param joints2D_heatmaps: (N, K, H, W) array of 2D joint heatmaps.
    :param eps: heatmap max threshold to count as detected joint.
    :return: joints2D: (N, K, 2) array of 2D joint coordinates.
             joints2D_vis: (N, K) bool array of 2D joint visibilties.
    """
    batch_size = joints2D_heatmaps.shape[0]
    num_joints = joints2D_heatmaps.shape[1]
    width = joints2D_heatmaps.shape[3]

    # Joints 2D given by max heatmap indices.
    # Since max and argmax are over batched 2D arrays, first flatten to 1D.
    max_vals_flat, max_indices_flat = torch.max(joints2D_heatmaps.view(batch_size, num_joints, -1),
                                                dim=-1)  # (N, K)
    # Convert 1D max indices to 2D max indices i.e. (x, y) coordinates.
    joints2D = torch.zeros(batch_size, num_joints, 2, device=joints2D_heatmaps.device)  # (N, K, 2)
    joints2D[:, :, 0] = max_indices_flat % width  # x-coordinate
    joints2D[:, :, 1] = torch.floor(max_indices_flat / float(width))  # y-coordinate

    # If heatmap is 0 everywhere (i.e. max value = 0), then no 2D coordinates
    # should be returned for that heatmap (i.e. joints2D not visible).
    # Following returns 1 for heatmaps with visible 2D coordinates (max val > eps) and -1 for heatmaps without.
    joints2D_vis = max_vals_flat > eps
    joints2D[torch.logical_not(joints2D_vis)] = -1

    return joints2D, joints2D_vis

