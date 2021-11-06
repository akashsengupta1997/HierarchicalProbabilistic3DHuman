import torch
import numpy as np

from utils.label_conversions import TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP


def random_joints2D_deviation(joints2D, delta_j2d_dev_range=[-5, 5], delta_j2d_hip_dev_range=[-15, 15]):
    """
    joints2D: (bs, num joints, num joints)
    """
    hip_joints = [11, 12]
    other_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14,15, 16]
    batch_size = joints2D.shape[0]
    device = joints2D.device

    l, h = delta_j2d_dev_range
    delta_j2d_dev = (h - l) * torch.rand(batch_size, len(other_joints), 2, device=device) + l
    joints2D[:, other_joints, :] = joints2D[:, other_joints, :] + delta_j2d_dev

    l, h = delta_j2d_hip_dev_range
    delta_j2d_hip_dev_range = (h - l) * torch.rand(batch_size, len(hip_joints), 2, device=device) + l
    joints2D[:, hip_joints, :] = joints2D[:, hip_joints, :] + delta_j2d_hip_dev_range

    return joints2D


def random_remove_bodyparts(seg,
                            classes_to_remove,
                            probabilities_to_remove_classes,
                            joints2D_visib,
                            probability_to_remove_joints):
    """
    :param seg: (bs, wh, wh)
    :param joints2D_visib: (bs, 17)
    """
    assert len(classes_to_remove) == len(probabilities_to_remove_classes)

    batch_size = seg.shape[0]
    for i in range(len(classes_to_remove)):
        class_to_remove = classes_to_remove[i]
        prob_to_remove = probabilities_to_remove_classes[i]

        # Determine which samples to augment (by removing class_to_remove) in the batch
        rand_vec = np.random.rand(batch_size) < prob_to_remove
        samples_to_augment = seg[rand_vec].clone()

        samples_to_augment[samples_to_augment == class_to_remove] = 0
        seg[rand_vec] = samples_to_augment

        if joints2D_visib is not None:
            if class_to_remove in TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP.keys():
                joint_to_remove = TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP[class_to_remove]

                # Determine which samples with removed class_to_remove will also have joints removed
                rand_vec_joints = np.random.rand(batch_size) < probability_to_remove_joints
                rand_vec_joints = np.logical_and(rand_vec, rand_vec_joints)  # Samples with removed class_to_remove AND removed corresponding joints
                joints2D_visib[rand_vec_joints, joint_to_remove] = 0

    return seg, joints2D_visib


def random_remove_joints2D(joints2D_visib,
                           joints_to_remove,
                           probability_to_remove=0.1):
    batch_size = joints2D_visib.shape[0]
    for joint in joints_to_remove:
        rand_vec = np.random.rand(batch_size) < probability_to_remove
        joints2D_visib[rand_vec, joint] = 0

    return joints2D_visib


def random_swap_joints2D(joints_2D, joints_to_swap, swap_probability=0.1):
    """
    joints_2D: (bs, num joints, 2)
    joints_to_swap: list of tuples of pairs of joints to swap
    """
    batch_size = joints_2D.shape[0]
    for pair in joints_to_swap:
        # Determine which samples to augment in the batch
        rand_vec = np.random.rand(batch_size) < swap_probability
        samples_to_augment = joints_2D[rand_vec].clone()
        temp = joints_2D[rand_vec].clone()

        # Swap joints given by pair
        samples_to_augment[:, pair[0], :] = temp[:, pair[1], :]
        samples_to_augment[:, pair[1], :] = temp[:, pair[0], :]

        joints_2D[rand_vec] = samples_to_augment

    return joints_2D


def random_occlude_box(seg,
                       occlude_probability=0.2,
                       occlude_box_dim=32.):
    """
    seg: (bs, wh, wh)
    """
    batch_size = seg.shape[0]
    seg_wh = seg.shape[-1]
    seg_centre = seg_wh/2
    x_h, x_l = seg_centre - 0.3*seg_wh/2, seg_centre + 0.3*seg_wh/2
    y_h, y_l = seg_centre - 0.3*seg_wh/2, seg_centre + 0.3*seg_wh/2

    x = (x_h - x_l) * np.random.rand(batch_size) + x_l
    y = (y_h - y_l) * np.random.rand(batch_size) + y_l
    box_x1 = (x - occlude_box_dim / 2).astype(np.int16)
    box_x2 = (x + occlude_box_dim / 2).astype(np.int16)
    box_y1 = (y - occlude_box_dim / 2).astype(np.int16)
    box_y2 = (y + occlude_box_dim / 2).astype(np.int16)

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            seg[i, box_x1[i]:box_x2[i], box_y1[i]:box_y2[i]] = 0

    return seg


def random_occlude_bottom_half(seg,
                               joints2D,
                               joints2D_visib,
                               occlude_probability=0.05):
    batch_size = seg.shape[0]
    wh = seg.shape[1]

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            occlude_from = int(wh / 2.0) + np.random.randint(low=-int(wh / 5.), high=int(wh / 5.))
            seg[i, occlude_from:, :] = 0

            if joints2D is not None:
                joints2D_to_occlude = joints2D[i, :, 1] > occlude_from
                joints2D_visib[i, joints2D_to_occlude] = False

    return seg, joints2D, joints2D_visib


def random_occlude_top_half(seg,
                            joints2D,
                            joints2D_visib,
                            occlude_probability=0.05):
    batch_size = seg.shape[0]
    wh = seg.shape[1]

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            occlude_up_to = int(wh / 2.0) + np.random.randint(low=-int(wh / 5.), high=int(wh / 5.))
            seg[i, :occlude_up_to, :] = 0

            if joints2D is not None:
                joints2D_to_occlude = joints2D[i, :, 1] < occlude_up_to
                joints2D_visib[i, joints2D_to_occlude] = False

    return seg, joints2D, joints2D_visib


def random_occlude_vertical_half(seg,
                                 joints2D,
                                 joints2D_visib,
                                 occlude_probability=0.05):
    batch_size = seg.shape[0]
    wh = seg.shape[1]

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            occlude_up_to = int(wh / 2.0) + np.random.randint(low=-int(wh / 30.), high=int(wh / 30.))
            if np.random.rand() > 0.5:
                seg[i, :, :occlude_up_to] = 0
                if joints2D is not None:
                    joints2D_to_occlude = joints2D[i, :, 0] < occlude_up_to
            else:
                seg[i, :, occlude_up_to:] = 0
                if joints2D is not None:
                    joints2D_to_occlude = joints2D[i, :, 0] > occlude_up_to
            if joints2D is not None:
                joints2D_visib[i, joints2D_to_occlude] = False

    return seg, joints2D, joints2D_visib


def augment_proxy_representation(seg,
                                 joints2D,
                                 joints2D_visib,
                                 proxy_rep_augment_config):
    new_seg = seg.clone()
    new_joints2D = joints2D.clone()
    new_joints2D_visib = joints2D_visib.clone()

    # Body part removal
    new_seg, new_joints2D_visib = random_remove_bodyparts(seg=new_seg,
                                                          classes_to_remove=proxy_rep_augment_config.REMOVE_PARTS_CLASSES,
                                                          probabilities_to_remove_classes=proxy_rep_augment_config.REMOVE_PARTS_PROBS,
                                                          joints2D_visib=new_joints2D_visib,
                                                          probability_to_remove_joints=proxy_rep_augment_config.REMOVE_APPENDAGE_JOINTS_PROB)

    # Box occlusion
    new_seg = random_occlude_box(seg=new_seg,
                                 occlude_probability=proxy_rep_augment_config.OCCLUDE_BOX_PROB,
                                 occlude_box_dim=proxy_rep_augment_config.OCCLUDE_BOX_DIM)

    # 2D joint swapping
    new_joints2D = random_swap_joints2D(joints_2D=new_joints2D,
                                        joints_to_swap=proxy_rep_augment_config.JOINTS_TO_SWAP,
                                        swap_probability=proxy_rep_augment_config.JOINTS_SWAP_PROB)

    # 2D joint deviation
    new_joints2D = random_joints2D_deviation(joints2D=new_joints2D,
                                             delta_j2d_dev_range=proxy_rep_augment_config.DELTA_J2D_DEV_RANGE,
                                             delta_j2d_hip_dev_range=proxy_rep_augment_config.DELTA_J2D_DEV_RANGE)

    # 2D joint removal (i.e. make invisible)
    new_joints2D_visib = random_remove_joints2D(joints2D_visib=new_joints2D_visib,
                                                joints_to_remove=proxy_rep_augment_config.REMOVE_JOINTS_INDICES,
                                                probability_to_remove=proxy_rep_augment_config.REMOVE_JOINTS_PROB)

    # Occlude bottom/top/left halves of the body (but not the background - see rgb_augmentation.py)
    new_seg, new_joints2D, new_joints2D_visib = random_occlude_bottom_half(seg=new_seg,
                                                                           joints2D=new_joints2D,
                                                                           joints2D_visib=new_joints2D_visib,
                                                                           occlude_probability=proxy_rep_augment_config.OCCLUDE_BOTTOM_PROB)
    new_seg, new_joints2D, new_joints2D_visib = random_occlude_top_half(new_seg,
                                                                        new_joints2D,
                                                                        new_joints2D_visib,
                                                                        occlude_probability=proxy_rep_augment_config.OCCLUDE_TOP_PROB)
    new_seg, new_joints2D, new_joints2D_visib = random_occlude_vertical_half(new_seg,
                                                                             new_joints2D,
                                                                             new_joints2D_visib,
                                                                             occlude_probability=proxy_rep_augment_config.OCCLUDE_VERTICAL_PROB)

    return new_seg, new_joints2D, new_joints2D_visib


def random_extreme_crop(seg,
                        extreme_crop_probability=0.05):
    """

    :param seg:
    :param extreme_crop_probability:
    :return: new_seg: part segmentations with regions to extreme crop
    """
    remove_legs_classes = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                       device=seg.device,
                                       dtype=seg.dtype)  # Legs and feet
    remove_legs_arms_classes = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22],
                                            device=seg.device,
                                            dtype=seg.dtype)  # Legs, feet, arms hands

    batch_size = seg.shape[0]
    new_seg = seg.clone()

    # Determine which samples to extreme crop in the batch
    # Remove legs for all samples to extreme crop (by setting to 0 in seg)
    # Remove legs + arms for ~half samples to extreme crop (by setting to 0 in seg)
    rand_vec = torch.rand(batch_size)
    rand_vec_legs = rand_vec < extreme_crop_probability * 0.5
    rand_vec_legs_arms = torch.logical_and(rand_vec > extreme_crop_probability * 0.5,
                                           rand_vec < extreme_crop_probability)
    samples_to_extreme_crop_legs = new_seg[rand_vec_legs]
    samples_to_extreme_crop_legs_arms = new_seg[rand_vec_legs_arms]

    indices_extreme_crop_legs = (samples_to_extreme_crop_legs[..., None] == remove_legs_classes).any(-1)
    samples_to_extreme_crop_legs[indices_extreme_crop_legs] = 0

    indices_extreme_crop_legs_arms = (samples_to_extreme_crop_legs_arms[..., None] == remove_legs_arms_classes).any(-1)
    samples_to_extreme_crop_legs_arms[indices_extreme_crop_legs_arms] = 0

    new_seg[rand_vec_legs] = samples_to_extreme_crop_legs
    new_seg[rand_vec_legs_arms] = samples_to_extreme_crop_legs_arms

    return new_seg



