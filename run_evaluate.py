import os
import numpy as np
import torch
import argparse

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults

from data.pw3d_eval_dataset import PW3DEvalDataset
from data.ssp3d_eval_dataset import SSP3DEvalDataset

from models.smpl_official import SMPL
from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.canny_edge_detector import CannyEdgeDetector
from evaluate.evaluate_poseMF_shapeGaussian_net import evaluate_pose_MF_shapeGaussian_net


def run_evaluate(device,
                 dataset_name,
                 pose_shape_weights_path,
                 pose_shape_cfg_path=None,
                 num_samples_for_metrics=10):

    # ------------------ Models ------------------
    # Config
    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()
    if pose_shape_cfg_path is not None:
        pose_shape_cfg.merge_from_file(pose_shape_cfg_path)
        print('\nLoaded Distribution Predictor config from', pose_shape_cfg_path)
    else:
        print('\nUsing default Distribution Predictor config.')

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL neutral/male/female models
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()
    smpl_model_male = SMPL(paths.SMPL,
                           batch_size=1,
                           gender='male').to(device)
    smpl_model_female = SMPL(paths.SMPL,
                             batch_size=1,
                             gender='female').to(device)

    # 3D shape and pose distribution predictor
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                   config=pose_shape_cfg).to(device)
    checkpoint = torch.load(pose_shape_weights_path, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('\nLoaded Distribution Predictor weights from', pose_shape_weights_path)

    # ------------------ Dataset + Metrics ------------------
    if dataset_name == '3dpw':
        metrics = ['PVE', 'PVE-SC', 'PVE-PA', 'PVE-T-SC', 'MPJPE', 'MPJPE-SC', 'MPJPE-PA']
        metrics += [metric + '_samples_min' for metric in metrics]
        save_path = './3dpw_eval'
        eval_dataset = PW3DEvalDataset(pw3d_dir_path=paths.PW3D_PATH,
                                       config=pose_shape_cfg,
                                       visible_joints_threshold=0.6)

    elif dataset_name == 'ssp3d':
        metrics = ['PVE-PA', 'PVE-T-SC', 'silhouette-IOU', 'joints2D-L2E', 'joints2Dsamples-L2E', 'silhouettesamples-IOU']
        save_path = './ssp3d_eval'
        eval_dataset = SSP3DEvalDataset(ssp3d_dir_path=paths.SSP3D_PATH,
                                        config=pose_shape_cfg)

    print("\nEvaluating on {} with {} eval examples.".format(dataset_name, str(len(eval_dataset))))
    print("Metrics:", metrics)
    print("Saving to:", save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ------------------ Evaluate ------------------
    torch.manual_seed(0)
    np.random.seed(0)
    evaluate_pose_MF_shapeGaussian_net(pose_shape_model=pose_shape_dist_model,
                                       pose_shape_cfg=pose_shape_cfg,
                                       smpl_model=smpl_model,
                                       smpl_model_male=smpl_model_male,
                                       smpl_model_female=smpl_model_female,
                                       edge_detect_model=edge_detect_model,
                                       device=device,
                                       eval_dataset=eval_dataset,
                                       metrics=metrics,
                                       save_path=save_path,
                                       num_samples_for_metrics=num_samples_for_metrics,
                                       sample_on_cpu=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, choices=['3dpw', 'ssp3d'])
    parser.add_argument('--pose_shape_weights', '-W3D', type=str, default='./model_files/poseMF_shapeGaussian_net_weights.tar')
    parser.add_argument('--pose_shape_cfg', type=str, default=None)
    parser.add_argument('--num_samples', '-N', type=int, default=10, help='Number of samples to use for sample-based evaluation metrics.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    run_evaluate(device=device,
                 dataset_name=args.dataset,
                 pose_shape_weights_path=args.pose_shape_weights,
                 pose_shape_cfg_path=args.pose_shape_cfg,
                 num_samples_for_metrics=args.num_samples)



