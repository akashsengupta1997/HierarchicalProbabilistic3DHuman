"""
Canny edge detection adapted from https://github.com/DCurro/CannyEdgePytorch
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.signal.windows import gaussian


class CannyEdgeDetector(nn.Module):
    def __init__(self,
                 non_max_suppression=True,
                 gaussian_filter_std=1.0,
                 gaussian_filter_size=5,
                 threshold=0.2):
        super(CannyEdgeDetector, self).__init__()

        self.threshold = threshold
        self.non_max_suppression = non_max_suppression

        # Gaussian filter for smoothing
        gaussian_filter = gaussian(gaussian_filter_size, std=gaussian_filter_std).reshape([1, gaussian_filter_size])
        gaussian_filter = gaussian_filter / gaussian_filter.sum()
        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1,
                                                    out_channels=1,
                                                    kernel_size=(1, gaussian_filter_size),
                                                    padding=(0, gaussian_filter_size // 2),
                                                    bias=False)
        # self.gaussian_filter_horizontal.weight[:] = torch.from_numpy(gaussian_filter).float()
        self.gaussian_filter_horizontal.weight.data = torch.from_numpy(gaussian_filter).float()[None, None, :, :]
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1,
                                                  out_channels=1,
                                                  kernel_size=(gaussian_filter_size, 1),
                                                  padding=(gaussian_filter_size // 2, 0),
                                                  bias=False)
        # self.gaussian_filter_vertical.weight[:] = torch.from_numpy(gaussian_filter.T)
        self.gaussian_filter_vertical.weight.data = torch.from_numpy(gaussian_filter.T).float()[None, None, :, :]

        # Sobel filter for gradient
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1,
                                                 out_channels=1,
                                                 kernel_size=sobel_filter.shape,
                                                 padding=sobel_filter.shape[0] // 2,
                                                 bias=False)
        # self.sobel_filter_horizontal.weight[:] = torch.from_numpy(sobel_filter).float()
        self.sobel_filter_horizontal.weight.data = torch.from_numpy(sobel_filter).float()[None, None, :, :]
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1,
                                               out_channels=1,
                                               kernel_size=sobel_filter.shape,
                                               padding=sobel_filter.shape[0] // 2,
                                               bias=False)
        # self.sobel_filter_vertical.weight[:] = torch.from_numpy(sobel_filter.T).float()
        self.sobel_filter_vertical.weight.data = torch.from_numpy(sobel_filter.T).float()[None, None, :, :]


        # Directional filters for non-max suppression (edge thinning) using gradient orientations.
        # filters were flipped manually
        if self.non_max_suppression:
            filter_0 = np.array([[0, 0, 0],
                                 [0, 1, -1],
                                 [0, 0, 0]])

            filter_45 = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -1]])

            filter_90 = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, -1, 0]])

            filter_135 = np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [-1, 0, 0]])

            filter_180 = np.array([[0, 0, 0],
                                   [-1, 1, 0],
                                   [0, 0, 0]])

            filter_225 = np.array([[-1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            filter_270 = np.array([[0, -1, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            filter_315 = np.array([[0, 0, -1],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])
            self.directional_filter = nn.Conv2d(in_channels=1,
                                                out_channels=8,
                                                kernel_size=filter_0.shape,
                                                padding=filter_0.shape[-1] // 2,
                                                bias=False)
            # self.directional_filter.weight[:] = torch.from_numpy(all_filters[:, None, ...])
            self.directional_filter.weight.data = torch.from_numpy(all_filters[:, None, :, :]).float()

    def forward(self, img):
        """
        :param img: (batch_size, num_channels, img_wh, img_wh)
        :return:
        """
        batch_size = img.shape[0]
        num_channels = img.shape[1]

        blurred_img = torch.zeros_like(img)  # (batch_size, num_channels, img_wh, img_wh)
        grad_x = torch.zeros((batch_size, 1, *img.shape[2:]), device=img.device)  # (batch_size, 1, img_wh, img_wh)
        grad_y = torch.zeros((batch_size, 1, *img.shape[2:]), device=img.device)  # (batch_size, 1, img_wh, img_wh)
        for c in range(num_channels):
            # Gaussian smoothing
            blurred = self.gaussian_filter_vertical(self.gaussian_filter_horizontal(img[:, [c], :, :]))   # (batch_size, 1, img_wh, img_wh)
            blurred_img[:, [c]] = blurred

            # Gradient
            grad_x += self.sobel_filter_horizontal(blurred)  # (batch_size, 1, img_wh, img_wh)
            grad_y += self.sobel_filter_vertical(blurred)  # (batch_size, 1, img_wh, img_wh)

        # Gradient magnitude and orientation
        grad_x, grad_y = grad_x / num_channels, grad_y / num_channels  # Average per-pixel gradients over channels
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5  # Per-pixel gradient magnitude
        grad_orientation = torch.atan2(grad_y, grad_x) * (180.0/np.pi) + 180.0  # Per-pixel gradient orientation in degrees with range (0°, 360°)
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0  # Bin gradient orientations

        # Thresholding
        thresholded_grad_magnitude = grad_magnitude.clone()
        thresholded_grad_magnitude[grad_magnitude < self.threshold] = 0.0

        output = {'blurred_img': blurred_img,  # (batch_size, num_channels, img_wh, img_wh)
                  'grad_magnitude': grad_magnitude,  # (batch_size, 1, img_wh, img_wh)
                  'grad_orientation': grad_orientation,  # (batch_size, 1, img_wh, img_wh)
                  'thresholded_grad_magnitude': thresholded_grad_magnitude}  # (batch_size, 1, img_wh, img_wh)
        assert grad_magnitude.size() == grad_orientation.size() == thresholded_grad_magnitude.size()

        # Non-max suppression (edge thinning)
        if self.non_max_suppression:
            all_direction_filtered = self.directional_filter(grad_magnitude)  # (batch_size, 8, img_wh, img_wh)
            positive_idx = (grad_orientation / 45) % 8  # (batch_size, 1, img_wh, img_wh)  Index of positive gradient direction (0: 0°, ..., 7: 315°) at each pixel
            thin_edges = grad_magnitude.clone()  # (batch_size, 1, img_wh, img_wh)
            for pos_i in range(4):
                neg_i = pos_i + 4
                is_oriented_i = (positive_idx == pos_i) * 1
                is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1  # > 0 if pixel is oriented in pos_i or neg_i direction
                pos_directional = all_direction_filtered[:, pos_i]
                neg_directional = all_direction_filtered[:, neg_i]
                selected_direction = torch.stack([pos_directional, neg_directional])

                # get the local maximum pixels for the angle
                is_max = selected_direction.min(dim=0)[0] > 0.0  # Check if pixel greater than neighbours in pos_i and neg_i directions.
                is_max = torch.unsqueeze(is_max, dim=1)

                # apply non maximum suppression
                to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
                thin_edges[to_remove] = 0.0
            thresholded_thin_edges = thin_edges.clone()
            thresholded_thin_edges[thin_edges < self.threshold] = 0.0

            output['thin_edges'] = thin_edges
            output['thresholded_thin_edges'] = thresholded_thin_edges

        return output
