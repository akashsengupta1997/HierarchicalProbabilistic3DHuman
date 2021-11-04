import numpy as np
import torch


def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints


def check_joints2d_visibility_torch(joints2d,
                                    img_wh,
                                    visibility=None):
    """
    Checks if 2D joints are within the image dimensions.
    """
    if visibility is None:
        visibility = torch.ones(joints2d.shape[:2], device=joints2d.device, dtype=torch.bool)
    visibility[joints2d[:, :, 0] > img_wh] = 0
    visibility[joints2d[:, :, 1] > img_wh] = 0
    visibility[joints2d[:, :, 0] < 0] = 0
    visibility[joints2d[:, :, 1] < 0] = 0

    return visibility


def check_joints2d_occluded_torch(seg14part, vis, pixel_count_threshold=50):
    """
    Check if 2D joints are not self-occluded in the rendered silhouette/seg, by checking if corresponding body parts are
    visible in the corresponding 14 part seg.
    :param seg14part: (B, D, D)
    :param vis: (B, 17)
    """
    new_vis = vis.clone()
    joints_to_check_and_corresponding_bodyparts = {7: 3, 8: 5, 9: 12, 10: 11, 13: 7, 14: 9, 15: 14, 16: 13}

    for joint_index in joints_to_check_and_corresponding_bodyparts.keys():
        part = joints_to_check_and_corresponding_bodyparts[joint_index]
        num_pixels_part = (seg14part == part).sum(dim=(1, 2))  # (B,)
        visibility_flag = (num_pixels_part > pixel_count_threshold)  # (B,)
        new_vis[:, joint_index] = (vis[:, joint_index] & visibility_flag)

    return new_vis

