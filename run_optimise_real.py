import os
import torch
import torchvision
import numpy as np
import argparse

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector

from renderers.pytorch3d_silh_opt_renderer import SilhouetteRenderer

from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths

from optimise.optim_real_data import optim_real_data
#from optimise.optim_real_data_clean import optim_real_data


def run_predict(device,
                image_dir,
                save_dir,
                pose_shape_weights_path,
                pose2D_hrnet_weights_path,
                pose_shape_cfg_path=None,
                already_cropped_images=False,
                visualise_samples=False,
                visualise_uncropped=False,
                joints2Dvisib_threshold=0.75,
                gender='neutral',
                verbose=False):

    # ------------------------- Models -------------------------
    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()
    if pose_shape_cfg_path is not None:
        pose_shape_cfg.merge_from_file(pose_shape_cfg_path)
        print('\nLoaded Distribution Predictor config from', pose_shape_cfg_path)
    else:
        print('\nUsing default Distribution Predictor config.')

    # Bounding box / Object detection model
    if not already_cropped_images:
        object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    else:
        object_detect_model = None

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(pose2D_hrnet_weights_path, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nLoaded HRNet weights from', pose2D_hrnet_weights_path)

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL model
    print('\nUsing {} SMPL model with {} shape parameters.'.format(gender, str(pose_shape_cfg.MODEL.NUM_SMPL_BETAS)))
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      gender=gender,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()

    # 3D shape and pose distribution predictor
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                   config=pose_shape_cfg).to(device)
    checkpoint = torch.load(pose_shape_weights_path, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('\nLoaded Distribution Predictor weights from', pose_shape_weights_path)



    img_wh = 256
    num_simul_subjects = 16
    num_imgs_per_subject = 2  # All subjects MUST have the SAME number of images
    init_foc_len = 240.
    blend_sigma = 1e-6
    blend_gamma = 1e-5
    num_iters = 1101
    shape_iters = 340
    meas_iters = 340  # If meas_iters == shape_iters, not doing any shape only optimisation
    lr = 0.0015
    silh_weight_sched = {'joint': 13., 'shape_only': 25., 'meas_only': 25.}
    joints2D_weight_sched = {'joint': 0.1, 'shape_only': 0.02, 'meas_only': 0.02}


    num_simul_imgs = 1
    smpl_faces = torch.from_numpy(smpl_model.faces[None].astype(np.int32)).to(device).expand(num_simul_imgs, -1, -1)
    silh_opt_renderer = SilhouetteRenderer(device=device,
                                        batch_size=num_simul_imgs,
                                        smpl_faces=smpl_faces,
                                        img_wh=img_wh,
                                        blend_gamma=blend_gamma,
                                        blend_sigma=blend_sigma,
                                        blur_radius=None,
                                        faces_per_pixel=8,
                                        bin_size=32,
                                        projection_type='perspective')
    

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    optim_real_data(pose_shape_model=pose_shape_dist_model,
                                     pose_shape_cfg=pose_shape_cfg,
                                     smpl_model=smpl_model,
                                     hrnet_model=hrnet_model,
                                     hrnet_cfg=pose2D_hrnet_cfg,
                                     edge_detect_model=edge_detect_model,
                                     device=device,
                                     image_dir=image_dir,
                                     save_dir=save_dir,
                                     object_detect_model=object_detect_model,
                                     joints2Dvisib_threshold=joints2Dvisib_threshold,
                                     visualise_uncropped=visualise_uncropped,
                                     visualise_samples=visualise_samples,
                                     silh_opt_renderer=silh_opt_renderer,
                                     verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-I', type=str, help='Path to directory of test images.', default="./my_data/images/")
    parser.add_argument('--save_dir', '-S', type=str, help='Path to directory where test outputs will be saved.')
    parser.add_argument('--pose_shape_weights', '-W3D', type=str, default='/scratches/nazgul/cq244/hkpd_depth/model_files/poseMF_shapeGaussian_net_weights_male.tar')
    parser.add_argument('--pose_shape_cfg', type=str, default=None)
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str, default='/scratches/nazgul/cq244/hkpd_depth/model_files/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--cropped_images', '-C', action='store_true', help='Images already cropped and centred.')
    parser.add_argument('--visualise_samples', '-VS', action='store_true')
    parser.add_argument('--visualise_uncropped', '-VU', action='store_true')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--gender', '-G', type=str, default='neutral', choices=['neutral', 'male', 'female'], help='Gendered SMPL models may be used.')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    run_predict(device=device,
                image_dir=args.image_dir,
                save_dir=args.save_dir,
                pose_shape_weights_path=args.pose_shape_weights,
                pose_shape_cfg_path=args.pose_shape_cfg,
                pose2D_hrnet_weights_path=args.pose2D_hrnet_weights,
                already_cropped_images=args.cropped_images,
                visualise_samples=args.visualise_samples,
                visualise_uncropped=args.visualise_uncropped,
                joints2Dvisib_threshold=args.joints2Dvisib_threshold,
                gender=args.gender,
                verbose=args.verbose)


