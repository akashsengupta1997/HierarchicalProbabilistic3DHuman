import torch
from torchvision import transforms

from utils.image_utils import convert_bbox_corners_to_centre_hw_torch, batch_crop_pytorch_affine


def get_kp_locations_confs_from_heatmaps(batch_heatmaps):
    """
    :param batch_heatmaps: (B, K, heatmap_height, heatmap_width)
    :return: pred_kps: (B, K, 2)
    :return: max_confs: (B, K)
    """
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # Argmax is easier to do over 1D axes/array, hence the below reshape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)  # (bs, num joints, height * width)
    max_confs, max_indices = torch.max(heatmaps_reshaped, dim=2)

    # Get 2D keypoint indices (i.e. locations) from 1D argmax indices
    pred_kps = torch.zeros(batch_size, num_joints, 2,
                           device=batch_heatmaps.device, dtype=torch.float32)
    pred_kps[:, :, 0] = max_indices % width
    pred_kps[:, :, 1] = torch.floor(max_indices / float(width))

    pred_mask = (max_confs > 0.0)[:, :, None]
    pred_kps *= pred_mask

    return pred_kps, max_confs


def predict_hrnet(hrnet_model,
                  hrnet_config,
                  image,
                  object_detect_model=None,
                  object_detect_threshold=0.8,
                  bbox_scale_factor=1.2):
    """
    :param hrnet_model:
    :param object_detect_model:
    :param image: (3, H, W) tensor, input RGB image
    :return: joints2D: (K, 2) tensor, 2D joint locations predicted by HRNet
    :return: joints2Dconfs: (K,) tensor, 2D joint confidences predicted by HRNet
    :return: bbox_centre, bbox_height, bbox_width: bounding box centre, height and width

    """
    image_height, image_width = image.shape[1:]
    if object_detect_model is not None:
        # Detecting object bounding boxes in input image
        # Bounding boxes are in (hor, vert) coordinates
        object_pred = object_detect_model(image[None, :, :, :])[0]

        # Select person bounding boxes with score > object detect threshold.
        pred_human_boxes = object_pred['boxes'][object_pred['labels'] == 1]  # 1 is COCO index for 'person' class
        pred_human_scores = object_pred['scores'][object_pred['labels'] == 1]
        pred_human_boxes = pred_human_boxes[pred_human_scores > object_detect_threshold]  # (num persons, 4)

        # Convert box corners to (centre, height, width) and select centre-most person box
        all_pred_centres, all_pred_heights, all_pred_widths = convert_bbox_corners_to_centre_hw_torch(
            bbox_corners=pred_human_boxes[:, [1, 0, 3, 2]])  # Use (vert, hor) coordinates
        if pred_human_boxes.shape[0] > 1:
            centre_dists = (all_pred_centres[:, 0] - image_height/2.0) ** 2 + (all_pred_centres[:, 1] - image_width/2.0) ** 2
            pred_centre = all_pred_centres[torch.argmin(centre_dists), :]
            pred_height = all_pred_heights[torch.argmin(centre_dists)]
            pred_width = all_pred_widths[torch.argmin(centre_dists)]
        else:
            try:
                pred_centre = all_pred_centres[0]
                pred_height = all_pred_heights[0]
                pred_width = all_pred_widths[0]
            except IndexError:
                print("Could not find person bounding box - using entire image!")
                pred_centre = torch.tensor(image.shape[1:], device=image.device, dtype=torch.float32) * 0.5
                pred_height = torch.tensor(image_height, device=image.device, dtype=torch.float32)
                pred_width = torch.tensor(image_width, device=image.device, dtype=torch.float32)
    else:
        pred_centre = torch.tensor(image.shape[1:], device=image.device, dtype=torch.float32) * 0.5
        pred_height = torch.tensor(image_height, device=image.device, dtype=torch.float32)
        pred_width = torch.tensor(image_width, device=image.device, dtype=torch.float32)

    # Convert box to be same aspect ratio as HrNet input
    aspect_ratio = float(hrnet_config.MODEL.IMAGE_SIZE[1]) / float(hrnet_config.MODEL.IMAGE_SIZE[0])
    if pred_height > pred_width * aspect_ratio:
        pred_width = pred_height / aspect_ratio
    elif pred_height < pred_width * aspect_ratio:
        pred_height = pred_width * aspect_ratio

    # Crop input image to centre-most person box + resize to 384x288 for HRNet input.
    image = batch_crop_pytorch_affine(input_wh=(image_width, image_height),
                                      output_wh=(hrnet_config.MODEL.IMAGE_SIZE[0], hrnet_config.MODEL.IMAGE_SIZE[1]),
                                      num_to_crop=1,
                                      device=image.device,
                                      rgb=image[None, :, :, :],
                                      bbox_centres=pred_centre[None, :],
                                      bbox_heights=pred_height[None],
                                      bbox_widths=pred_width[None],
                                      orig_scale_factor=bbox_scale_factor)['rgb'][0]  # (3, 384, 288)

    # Predict 2D joint heatmaps using HRNet
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    pred_heatmaps = hrnet_model(transform(image)[None, :, :, :])  # (1, 17, 96, 72)
    pred_joints2D, pred_joints2Dconfs = get_kp_locations_confs_from_heatmaps(pred_heatmaps)

    # Rescale 2D joint locations back to HRNet input size
    pred_joints2D *= hrnet_config.MODEL.IMAGE_SIZE[0] / hrnet_config.MODEL.HEATMAP_SIZE[0]

    output = {'joints2D': pred_joints2D[0],
              'joints2Dconfs': pred_joints2Dconfs[0],
              'cropped_image': image,
              'bbox_centre': pred_centre,
              'bbox_height': pred_height,
              'bbox_width': pred_width}

    return output



