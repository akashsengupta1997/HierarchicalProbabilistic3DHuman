import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps


class PW3DEvalDataset(Dataset):
    def __init__(self,
                 pw3d_dir_path,
                 config,
                 visible_joints_threshold=None):
        super(PW3DEvalDataset, self).__init__()

        self.cropped_frames_dir = os.path.join(pw3d_dir_path, 'cropped_frames')

        data = np.load(os.path.join(pw3d_dir_path, '3dpw_test.npz'))
        self.frame_fnames = data['imgname']
        self.pose = data['pose']
        self.shape = data['shape']
        self.gender = data['gender']

        self.keypoints = np.load(os.path.join(pw3d_dir_path, 'hrnet_results_centred.npy'))

        self.img_wh = config.DATA.PROXY_REP_SIZE
        self.hmaps_gaussian_std = config.DATA.HEATMAP_GAUSSIAN_STD
        self.visible_joints_threshold = visible_joints_threshold

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # ---------------------- Inputs ----------------------
        fname = self.frame_fnames[index]
        cropped_frame_path = os.path.join(self.cropped_frames_dir, fname)

        image = cv2.cvtColor(cv2.imread(cropped_frame_path), cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        assert (orig_height == orig_width), "Resizing non-square image to square will cause unwanted stretching/squeezing!"
        image = cv2.resize(image, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, [2, 0, 1]) / 255.0

        keypoints = self.keypoints[index]
        keypoints_confidence = keypoints[:, 2]  # (17,)
        keypoints = keypoints[:, :2]
        keypoints = keypoints * np.array([self.img_wh / float(orig_width),
                                          self.img_wh / float(orig_height)])
        heatmaps = convert_2Djoints_to_gaussian_heatmaps(keypoints.round().astype(np.int16),
                                                         self.img_wh,
                                                         std=self.hmaps_gaussian_std)
        if self.visible_joints_threshold is not None:
            keypoints_visiblity_flag = keypoints_confidence > self.visible_joints_threshold
            keypoints_visiblity_flag[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
            heatmaps = heatmaps * keypoints_visiblity_flag[None, None, :]
        heatmaps = np.transpose(heatmaps, [2, 0, 1])

        # ---------------------- Targets ----------------------
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

        image = torch.from_numpy(image).float()
        heatmaps = torch.from_numpy(heatmaps).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        return {'image': image,
                'heatmaps': heatmaps,
                'pose': pose,
                'shape': shape,
                'fname': fname,
                'gender': gender}
