import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps
from utils.image_utils import batch_crop_opencv_affine


class SSP3DEvalDataset(Dataset):
    def __init__(self,
                 ssp3d_dir_path,
                 config,
                 visible_joints_threshold=None):
        super(SSP3DEvalDataset, self).__init__()

        self.images_dir = os.path.join(ssp3d_dir_path, 'images')
        self.silhouettes_dir = os.path.join(ssp3d_dir_path, 'silhouettes')

        data = np.load(os.path.join(ssp3d_dir_path, 'labels.npz'))
        self.frame_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']
        self.keypoints = data['joints2D']
        self.bbox_centres = data['bbox_centres']  # Tight bounding box centre
        self.bbox_whs = data['bbox_whs']  # Tight bounding box width/height
        self.genders = data['genders']

        self.img_wh = config.DATA.PROXY_REP_SIZE
        self.hmaps_gaussian_std = config.DATA.HEATMAP_GAUSSIAN_STD
        self.bbox_scale_factor = config.DATA.BBOX_SCALE_FACTOR
        self.visible_joints_threshold = visible_joints_threshold

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # ------------------ Inputs ------------------
        fname = self.frame_fnames[index]
        image_path = os.path.join(self.images_dir, fname)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        keypoints = np.copy(self.keypoints[index])
        keypoints_conf = keypoints[:, 2]  # (17,)

        # Images + Joints need to be cropped to bounding box.
        bbox_centre = self.bbox_centres[index]
        bbox_wh = self.bbox_whs[index]
        crop_outputs = batch_crop_opencv_affine(output_wh=(self.img_wh, self.img_wh),
                                                num_to_crop=1,
                                                rgb=image[None].transpose(0, 3, 1, 2),
                                                joints2D=keypoints[None, :, :2],
                                                bbox_centres=bbox_centre[None],
                                                bbox_whs=[bbox_wh],
                                                orig_scale_factor=self.bbox_scale_factor)
        image = crop_outputs['rgb'][0] / 255.0

        keypoints = crop_outputs['joints2D'][0]
        heatmaps = convert_2Djoints_to_gaussian_heatmaps(keypoints.astype(np.int16), self.img_wh,
                                                         std=self.hmaps_gaussian_std)
        if self.visible_joints_threshold is not None:
            keypoints_visibility_flag = keypoints_conf > self.visible_joints_threshold
            keypoints_visibility_flag[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
            heatmaps = heatmaps * keypoints_visibility_flag[None, None, :]
        heatmaps = np.transpose(heatmaps, [2, 0, 1])

        # ------------------ Targets ------------------
        shape = self.body_shapes[index]
        pose = self.body_poses[index]
        gender = self.genders[index]
        silhouette = cv2.imread(os.path.join(self.silhouettes_dir, fname), 0)
        silhouette = batch_crop_opencv_affine(output_wh=(self.img_wh, self.img_wh),
                                              num_to_crop=1,
                                              seg=silhouette[None],
                                              bbox_centres=bbox_centre[None],
                                              bbox_whs=[bbox_wh],
                                              orig_scale_factor=self.bbox_scale_factor)['seg'][0]

        image = torch.from_numpy(image).float()
        heatmaps = torch.from_numpy(heatmaps).float()
        shape = torch.from_numpy(shape).float()
        pose = torch.from_numpy(pose).float()

        return {'image': image,
                'heatmaps': heatmaps,
                'shape': shape,
                'pose': pose,
                'silhouette': silhouette,
                'keypoints': keypoints,
                'fname': fname,
                'gender': gender}
