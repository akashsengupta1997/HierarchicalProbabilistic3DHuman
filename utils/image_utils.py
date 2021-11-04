import cv2
import numpy as np
import torch
import torch.nn.functional as F


def convert_bbox_corners_to_centre_hw(bbox_corners):
    """
    Convert bbox coordinates from [top left:(x1, y1), bot right: (x2, y2)]  to centre, height, width.
    x = rows/vertical axis, y = columns/horizontal axis
    :param bbox_corners: (B, 4) tensor of bounding box corners [x1, y1, x2, y2] in (vertical, horizontal) coordinates.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    height = x2 - x1
    width = y2 - y1

    return centre, height, width


def convert_bbox_corners_to_centre_hw_torch(bbox_corners):
    """
    Convert bbox coordinates from [top left:(x1, y1), bot right: (x2, y2)]  to centre, height, width.
    x = rows/vertical axis, y = columns/horizontal axis
    :param bbox_corners: (B, 4) tensor of bounding box corners [x1, y1, x2, y2] in (vertical, horizontal) coordinates.
    :returns bbox_centres: (B, 2) tensor of bounding box centres in (vertical, horizontal) coordinates.
    :returns bbox_heights and bbox_widths: (B,) tensors of bounding box heights and widths
    """
    bbox_centres = torch.zeros(bbox_corners.shape[0], 2,
                               dtype=torch.float32, device=bbox_corners.device)
    bbox_centres[:, 0] = (bbox_corners[:, 0] + bbox_corners[:, 2]) / 2.0
    bbox_centres[:, 1] = (bbox_corners[:, 1] + bbox_corners[:, 3]) / 2.0
    bbox_heights = bbox_corners[:, 2] - bbox_corners[:, 0]
    bbox_widths = bbox_corners[:, 3] - bbox_corners[:, 1]

    return bbox_centres, bbox_heights, bbox_widths


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])


def batch_add_rgb_background(backgrounds,
                             rgb,
                             seg):
    """
    :param backgrounds: (bs, 3, wh, wh)
    :param rgb: (bs, 3, wh, wh)
    :param iuv: (bs, wh, wh)
    :return: rgb_with_background: (bs, 3, wh, wh)
    """
    background_pixels = seg[:, None, :, :] == 0  # Body pixels are > 0 and out of frame pixels are -1
    rgb_with_background = rgb * (torch.logical_not(background_pixels)) + backgrounds * background_pixels
    return rgb_with_background


def batch_crop_opencv_affine(output_wh,
                             num_to_crop,
                             iuv=None,
                             joints2D=None,
                             rgb=None,
                             seg=None,
                             bbox_centres=None,
                             bbox_heights=None,
                             bbox_widths=None,
                             bbox_whs=None,
                             joints2D_vis=None,
                             orig_scale_factor=1.2,
                             delta_scale_range=None,
                             delta_centre_range=None,
                             out_of_frame_pad_val=0,
                             solve_for_affine_trans=False,
                             uncrop=False,
                             uncrop_wh=None):
    """
    :param output_wh: tuple, output image (width, height)
    :param num_to_crop: scalar int, number of images in batch
    :param iuv: (B, 3, H, W)
    :param joints2D: (B, K, 2)
    :param rgb: (B, 3, H, W)
    :param seg: (B, H, W)
    :param bbox_centres: (B, 2), bounding box centres in (vertical, horizontal) coordinates
    :param bbox_heights: (B,)
    :param bbox_widths: (B,)
    :param bbox_whs: (B,) width/height for square bounding boxes
    :param joints2D_vis: (B, K)
    :param orig_scale_factor: original bbox scale factor (pre-augmentation)
    :param delta_scale_range: bbox scale augmentation range
    :param delta_centre_range: bbox centre augmentation range
    :param out_of_frame_pad_val: padding value for out-of-frame region after affine transform
    :param solve_for_affine_trans: bool, if true use cv2.getAffineTransform() to determine
        affine transformation matrix.
    :param uncrop: bool, if true uncrop image by applying inverse affine transformation
    :param uncrop_wh: tuple, output image size for uncropping.
    :return: cropped iuv/joints2D/rgb/seg, resized to output_wh
    """
    output_wh = np.array(output_wh, dtype=np.float32)
    cropped_dict = {}
    if iuv is not None:
        if not uncrop:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if joints2D is not None:
        cropped_dict['joints2D'] = np.zeros_like(joints2D)
    if rgb is not None:
        if not uncrop:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if seg is not None:
        if not uncrop:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)

    for i in range(num_to_crop):
        if bbox_centres is None:
            assert (iuv is not None) or (joints2D is not None) or (seg is not None), "Need either IUV, Seg or 2D Joints to determine bounding boxes!"
            if iuv is not None:
                # Determine bounding box corners from segmentation foreground/body pixels from IUV map
                body_pixels = np.argwhere(iuv[i, 0, :, :] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif seg is not None:
                # Determine bounding box corners from segmentation foreground/body pixels
                body_pixels = np.argwhere(seg[i] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif joints2D is not None:
                # Determine bounding box corners from 2D joints
                visible_joints2D = joints2D[i, joints2D_vis[i]]
                bbox_corners = np.concatenate([np.amin(visible_joints2D, axis=0)[::-1],   # (hor, vert) coords to (vert, hor) coords
                                               np.amax(visible_joints2D, axis=0)[::-1]])
                if (bbox_corners[:2] == bbox_corners[2:]).all():  # This can happen if only 1 joint is visible in input
                    print('Only 1 visible joint in input!')
                    bbox_corners[2:] = bbox_corners[:2] + output_wh / 4.
            bbox_centre, bbox_height, bbox_width = convert_bbox_corners_to_centre_hw(bbox_corners)
        else:
            bbox_centre = bbox_centres[i]
            if bbox_whs is not None:
                bbox_height = bbox_whs[i]
                bbox_width = bbox_whs[i]
            else:
                bbox_height = bbox_heights[i]
                bbox_width = bbox_widths[i]

        if not uncrop:
            # Change bounding box aspect ratio to match output aspect ratio
            aspect_ratio = output_wh[1] / output_wh[0]
            if bbox_height > bbox_width * aspect_ratio:
                bbox_width = bbox_height / aspect_ratio
            elif bbox_height < bbox_width * aspect_ratio:
                bbox_height = bbox_width * aspect_ratio

            # Scale bounding boxes + Apply random augmentations
            if delta_scale_range is not None:
                l, h = delta_scale_range
                delta_scale = (h - l) * np.random.rand() + l
                scale_factor = orig_scale_factor + delta_scale
            else:
                scale_factor = orig_scale_factor
            bbox_height = bbox_height * scale_factor
            bbox_width = bbox_width * scale_factor
            if delta_centre_range is not None:
                l, h = delta_centre_range
                delta_centre = (h - l) * np.random.rand(2) + l
                bbox_centre = bbox_centre + delta_centre

            # Determine affine transformation mapping bounding box to output image
            output_centre = output_wh * 0.5
            if solve_for_affine_trans:
                # Solve for affine transformation using 3 point correspondences (6 equations)
                bbox_points = np.zeros((3, 2), dtype=np.float32)
                bbox_points[0, :] = bbox_centre[::-1]  # (vert, hor) coordinates to (hor, vert coordinates)
                bbox_points[1, :] = bbox_centre[::-1] + np.array([bbox_width * 0.5, 0], dtype=np.float32)
                bbox_points[2, :] = bbox_centre[::-1] + np.array([0, bbox_height * 0.5], dtype=np.float32)

                output_points = np.zeros((3, 2), dtype=np.float32)
                output_points[0, :] = output_centre
                output_points[1, :] = output_centre + np.array([output_wh[0] * 0.5, 0], dtype=np.float32)
                output_points[2, :] = output_centre + np.array([0, output_wh[1] * 0.5], dtype=np.float32)
                affine_trans = cv2.getAffineTransform(bbox_points, output_points)
            else:
                # Hand-code affine transformation matrix - easy for cropping = scale + translate
                affine_trans = np.zeros((2, 3), dtype=np.float32)
                affine_trans[0, 0] = output_wh[0] / bbox_width
                affine_trans[1, 1] = output_wh[1] / bbox_height
                affine_trans[:, 2] = output_centre - (output_wh / np.array([bbox_width, bbox_height])) * bbox_centre[::-1]  # (vert, hor) coords to (hor, vert) coords
        else:
            # Hand-code inverse affine transformation matrix - easy for UN-cropping = scale + translate
            affine_trans = np.zeros((2, 3), dtype=np.float32)
            output_centre = output_wh * 0.5
            affine_trans[0, 0] = bbox_width / output_wh[0]
            affine_trans[1, 1] = bbox_height / output_wh[1]
            affine_trans[:, 2] = bbox_centre[::-1] - (np.array([bbox_width, bbox_height]) / output_wh) * output_centre

        # Apply affine transformation inputs.
        if iuv is not None:
            cropped_dict['iuv'][i] = cv2.warpAffine(src=iuv[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=out_of_frame_pad_val).transpose(2, 0, 1)
        if joints2D is not None:
            joints2D_homo = np.concatenate([joints2D[i], np.ones((joints2D.shape[1], 1))],
                                           axis=-1)
            cropped_dict['joints2D'][i] = np.einsum('ij,kj->ki', affine_trans, joints2D_homo)

        if rgb is not None:
            cropped_dict['rgb'][i] = cv2.warpAffine(src=rgb[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0).transpose(2, 0, 1)
        if seg is not None:
            cropped_dict['seg'][i] = cv2.warpAffine(src=seg[i],
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0)

    return cropped_dict


def batch_crop_pytorch_affine(input_wh,
                              output_wh,
                              num_to_crop,
                              device,
                              iuv=None,
                              joints2D=None,
                              rgb=None,
                              seg=None,
                              bbox_determiner=None,
                              bbox_centres=None,
                              bbox_heights=None,
                              bbox_widths=None,
                              joints2D_vis=None,
                              orig_scale_factor=1.2,
                              delta_scale_range=None,
                              delta_centre_range=None,
                              out_of_frame_pad_val=0):
    """
    :param input_wh: tuple, input image (width, height)
    :param output_wh: tuple, output image (width, height)
    :param num_to_crop: number of images in batch
    :param iuv: (B, 3, H, W)
    :param joints2D: (B, K, 2)
    :param rgb: (B, 3, H, W)
    :param seg: (B, H, W)
    :param bbox_determiner: (B, H, W) segmentation/silhouette used to determine bbox corners if bbox corners
                            not determined by given iuv/joints2D/seg (e.g. used for extreme_crop augmentation)
    :param bbox_centres: (B, 2) bounding box centres in (vertical, horizontal) coordinates
    :param bbox_heights: (B,)
    :param bbox_widths: (B,
    :param joints2D_vis: (B, K)
    :param orig_scale_factor: original bbox scale factor (pre-augmentation)
    :param delta_scale_range: bbox scale augmentation range
    :param delta_centre_range: bbox centre augmentation range
    :param out_of_frame_pad_val: padding value for out-of-frame region after affine transform
    :return: Given iuv/joints2D/rgb/seg inputs, crops around person bounding box in input,
             resizes to output_wh and returns.
             Cropping + resizing is done using Pytorch's affine_grid and grid_sampling
    """
    input_wh = torch.tensor(input_wh, device=device, dtype=torch.float32)
    output_wh = torch.tensor(output_wh, device=device, dtype=torch.float32)
    if bbox_centres is None:
        # Need to determine bounding box from given IUV/Seg/2D Joints
        bbox_corners = torch.zeros(num_to_crop, 4, dtype=torch.float32, device=device)
        for i in range(num_to_crop):
            if bbox_determiner is None:
                assert (iuv is not None) or (joints2D is not None) or (seg is not None), "Need either IUV, Seg or 2D Joints to determine bounding boxes!"
                if iuv is not None:
                    # Determine bounding box corners from segmentation foreground/body pixels from IUV map
                    body_pixels = torch.nonzero(iuv[i, 0, :, :] != 0, as_tuple=False)
                    bbox_corners[i, :2], _ = torch.min(body_pixels, dim=0)  # Top left
                    bbox_corners[i, 2:], _ = torch.max(body_pixels, dim=0)  # Bot right
                elif seg is not None:
                    # Determine bounding box corners from segmentation foreground/body pixels
                    body_pixels = torch.nonzero(seg[i] != 0, as_tuple=False)
                    bbox_corners[i, :2], _ = torch.min(body_pixels, dim=0)  # Top left
                    bbox_corners[i, 2:], _ = torch.max(body_pixels, dim=0)  # Bot right
                elif joints2D is not None:
                    # Determine bounding box corners from 2D joints
                    visible_joints2D = joints2D[i, joints2D_vis[i]]
                    bbox_corners[i, :2], _ = torch.min(visible_joints2D, dim=0)  # Top left
                    bbox_corners[i, 2:], _ = torch.max(visible_joints2D, dim=0)  # Bot right
                    bbox_corners[i] = bbox_corners[i, [1, 0, 3, 2]]  # (horizontal, vertical) coordinates to (vertical, horizontal coordinates)
                    if (bbox_corners[:2] == bbox_corners[2:]).all():  # This can happen if only 1 joint is visible in input
                        print('Only 1 visible joint in input!')
                        bbox_corners[2] = bbox_corners[0] + output_wh[1]
                        bbox_corners[3] = bbox_corners[1] + output_wh[0]
            else:
                # Determine bounding box corners using given "bbox determiner"
                body_pixels = torch.nonzero(bbox_determiner[i] != 0, as_tuple=False)
                bbox_corners[i, :2], _ = torch.min(body_pixels, dim=0)  # Top left
                bbox_corners[i, 2:], _ = torch.max(body_pixels, dim=0)  # Bot right

        bbox_centres, bbox_heights, bbox_widths = convert_bbox_corners_to_centre_hw_torch(bbox_corners)

    # Change bounding box aspect ratio to match output aspect ratio
    aspect_ratio = (output_wh[1] / output_wh[0]).item()
    bbox_widths[bbox_heights > bbox_widths * aspect_ratio] = bbox_heights[bbox_heights > bbox_widths * aspect_ratio] / aspect_ratio
    bbox_heights[bbox_heights < bbox_widths * aspect_ratio] = bbox_widths[bbox_heights < bbox_widths * aspect_ratio] * aspect_ratio

    # Scale bounding boxes + Apply random augmentations
    if delta_scale_range is not None:
        l, h = delta_scale_range
        delta_scale = (h - l) * torch.rand(num_to_crop, device=device, dtype=torch.float32) + l
        scale_factor = orig_scale_factor + delta_scale
    else:
        scale_factor = orig_scale_factor
    bbox_heights = bbox_heights * scale_factor
    bbox_widths = bbox_widths * scale_factor
    if delta_centre_range is not None:
        l, h = delta_centre_range
        delta_centre = (h - l) * torch.rand(num_to_crop, 2, device=device, dtype=torch.float32) + l
        bbox_centres = bbox_centres + delta_centre

    # Hand-code affine transformation matrix - easy for cropping = scale + translate
    output_centre = output_wh * 0.5
    affine_trans = torch.zeros(num_to_crop, 2, 3, dtype=torch.float32, device=device)
    affine_trans[:, 0, 0] = output_wh[0] / bbox_widths
    affine_trans[:, 1, 1] = output_wh[1] / bbox_heights
    bbox_whs = torch.stack([bbox_widths, bbox_heights], dim=-1)
    affine_trans[:, :, 2] = output_centre - (output_wh / bbox_whs) * bbox_centres[:, [1, 0]]  # (vert, hor) to (hor, vert)

    # Pytorch needs NORMALISED INVERSE (compared to openCV) affine transform matrix for pytorch grid sampling
    # Since transform is just scale + translate, it is faster to hand-code noramlise + inverse than to use torch.inverse()
    # Forward transform: unnormalise with input dimensions + affine transform + normalise with output dimensions
    # Pytorch affine input = (Forward transform)^-1
    # see https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/18
    affine_trans_inv_normed = torch.zeros(num_to_crop, 2, 3, dtype=torch.float32, device=device)
    affine_trans_inv_normed[:, 0, 0] = bbox_widths / input_wh[0]
    affine_trans_inv_normed[:, 1, 1] = bbox_heights / input_wh[1]
    affine_trans_inv_normed[:, :, 2] = -affine_trans[:, :, 2] / (output_wh / bbox_whs)
    affine_trans_inv_normed[:, :, 2] = affine_trans_inv_normed[:, :, 2] / (input_wh * 0.5) + (bbox_whs / input_wh) - 1

    # Apply affine transformation inputs.
    affine_grid = F.affine_grid(theta=affine_trans_inv_normed,
                                size=[num_to_crop, 1, int(output_wh[1]), int(output_wh[0])],
                                align_corners=False)

    cropped_dict = {}
    if iuv is not None:
        cropped_dict['iuv'] = F.grid_sample(input=iuv - out_of_frame_pad_val,
                                            grid=affine_grid,
                                            mode='nearest',
                                            padding_mode='zeros',
                                            align_corners=False) + out_of_frame_pad_val
    if joints2D is not None:
        joints2D_homo = torch.cat([joints2D,
                                   torch.ones(num_to_crop, joints2D.shape[1], 1, device=device, dtype=torch.float32)],
                                  dim=-1)
        cropped_dict['joints2D'] = torch.einsum('bij,bkj->bki', affine_trans, joints2D_homo)

    if rgb is not None:
        cropped_dict['rgb'] = F.grid_sample(input=rgb,
                                            grid=affine_grid,
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=False)
    if seg is not None:
        cropped_dict['seg'] = F.grid_sample(input=seg,
                                            grid=affine_grid,
                                            mode='nearest',
                                            padding_mode='zeros',
                                            align_corners=False)

    return cropped_dict

