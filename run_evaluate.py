import os
import torch
import argparse

from configs import paths
from configs.poseMF_shapeGaussian_net_config import config

from data.pw3d_eval_dataset import PW3DEvalDataset
from data.ssp3d_eval_dataset import SSP3DEvalDataset

from models.smpl_official import SMPL
from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.canny_edge_detector import CannyEdgeDetector
from evaluate.evaluate_poseMF_shapeGaussian_net import evaluate_pose_MF_shapeGaussian_net


def run_evaluate(device,
                 dataset_name,
                 num_samples_for_metrics=10):

    # ------------------ Models ------------------
    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=config.DATA.EDGE_NMS,
                                          gaussian_filter_std=config.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=config.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=config.DATA.EDGE_THRESHOLD).to(device)

    # SMPL neutral/male/female models
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      num_betas=config.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()
    smpl_model_male = SMPL(paths.SMPL,
                           batch_size=1,
                           gender='male').to(device)
    smpl_model_female = SMPL(config.SMPL_MODEL_DIR,
                             batch_size=1,
                             gender='female').to(device)

    # 3D shape and pose distribution predictor
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                   config=config).to(device)
    checkpoint = torch.load(paths.POSE_SHAPE_NET_WEIGHTS, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('Loaded Distribution Predictor weights from', paths.POSE_SHAPE_NET_WEIGHTS)

    # ------------------ Dataset + Metrics + Evaluate ------------------
    if dataset_name == '3dpw':
        metrics = ['pves', 'pves_sc', 'pves_pa', 'pve-ts_sc', 'mpjpes', 'mpjpes_sc', 'mpjpes_pa']
        metrics += [metric + '_samples_min' for metric in metrics]
        save_path = './3dpw_eval'
        eval_dataset = PW3DEvalDataset(pw3d_dir_path=paths.PW3D_PATH,
                                       config=config,
                                       visible_joints_threshold=0.6)

    elif dataset_name == 'ssp3d':
        metrics = ['pves_pa', 'pve-ts_sc', 'silhouette_ious', 'joints2D_l2es', 'joints2Dsamples_l2es', 'silhouettesamples_ious']
        save_path = './ssp3d_eval'
        eval_dataset = SSP3DEvalDataset(ssp3d_dir_path=paths.SSP3D_PATH,
                                        config=config)

    print("\nEval examples found:", len(eval_dataset))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    evaluate_pose_MF_shapeGaussian_net(pose_shape_model=pose_shape_dist_model,
                                       pose_shape_config=config,
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
    parser.add_argument('--num_samples', '-N', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    run_evaluate(device=device,
                 dataset_name=args.dataset,
                 num_samples_for_metrics=args.num_samples)



