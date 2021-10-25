import os
import torch
import torchvision
import numpy as np
import argparse

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector

from configs.poseMF_shapeGaussian_net_config import config
from configs.pose2D_hrnet_config import pose2D_hrnet_config
from configs import paths

from predict.predict_poseMF_shapeGaussian_net import predict_poseMF_shapeGaussian_net


def run_predict(device,
                image_dir,
                save_dir,
                visualise_samples=False,
                visualise_uncropped=False,
                joints2Dvisib_threshold=0.75):

    # ------------------------- Load Models -------------------------
    # Bounding box / Object detection model
    object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_config).to(device)
    hrnet_checkpoint = torch.load(paths.HRNET_WEIGHTS, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nLoaded HRNet weights from', paths.HRNET_WEIGHTS)

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=config.DATA.EDGE_NMS,
                                          gaussian_filter_std=config.DATA.GAUSSIAN_STD,
                                          gaussian_filter_size=config.DATA.GAUSSIAN_SIZE,
                                          threshold=config.DATA.EDGE_THRESHOLD).to(device)

    # SMPL model
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      num_betas=config.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()

    # 3D shape and pose distribution predictor
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                   config=config).to(device)
    checkpoint = torch.load(paths.POSE_SHAPE_NET_WEIGHTS, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('Loaded Distribution Predictor weights from', paths.POSE_SHAPE_NET_WEIGHTS)

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    predict_poseMF_shapeGaussian_net(pose_shape_model=pose_shape_dist_model,
                                     pose_shape_config=config,
                                     smpl_model=smpl_model,
                                     hrnet_model=hrnet_model,
                                     hrnet_config=pose2D_hrnet_config,
                                     object_detect_model=object_detect_model,
                                     edge_detect_model=edge_detect_model,
                                     device=device,
                                     image_dir=image_dir,
                                     save_dir=save_dir,
                                     joints2Dvisib_threshold=joints2Dvisib_threshold,
                                     visualise_uncropped=visualise_uncropped,
                                     visualise_samples=visualise_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-I', type=str)
    parser.add_argument('--save_dir', '-S', type=str)
    parser.add_argument('--visualise_samples', '-VS', action='store_true')
    parser.add_argument('--visualise_uncropped', '-VU', action='store_true')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    run_predict(device=device,
                image_dir=args.image_dir,
                save_dir=args.save_dir,
                visualise_samples=args.visualise_samples,
                visualise_uncropped=args.visualise_uncropped,
                joints2Dvisib_threshold=args.joints2Dvisib_threshold)



