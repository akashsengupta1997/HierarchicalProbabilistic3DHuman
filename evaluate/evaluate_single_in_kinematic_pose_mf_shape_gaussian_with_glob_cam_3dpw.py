import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from smplx.lbs import batch_rodrigues

from models.smpl_official import SMPL

from metrics.eval_metrics_tracker import EvalMetricsTracker

from utils.rigid_transform_utils import rot6d_to_rotmat, rotate_global_pose_rotmats_torch
from utils.sampling_utils import pose_matrix_fisher_sampling_torch
from utils.label_conversions import ALL_JOINTS_TO_H36M_MAP, H36M_TO_J14


def evaluate_pose_MF_shapeGaussian_with_glob_cam_3dpw(model,
                                                      eval_dataset,
                                                      metrics_to_track,
                                                      device,
                                                      save_path,
                                                      num_workers=4,
                                                      pin_memory=True,
                                                      visualise=True,
                                                      save_per_frame_metrics=True,
                                                      edge_detector=None,
                                                      edge_nms=False,
                                                      num_smpl_betas=10,
                                                      num_samples_for_metrics=10,
                                                      sample_on_cpu=False):
    """
    Evaluator for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on 3DPW.
    Input --> ResNet --> image features --> FC layers --> MF over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl = SMPL(config.SMPL_300_MODEL_DIR, batch_size=1, num_betas=num_smpl_betas)
    smpl_male = SMPL(config.SMPL_300_MODEL_DIR, batch_size=1, gender='male', num_betas=10)
    smpl_female = SMPL(config.SMPL_300_MODEL_DIR, batch_size=1, gender='female', num_betas=10)
    smpl.to(device)
    smpl_male.to(device)
    smpl_female.to(device)

    # Instantiate metrics tracker
    metrics_tracker = EvalMetricsTracker(metrics_to_track,
                                         save_path=save_path,
                                         save_per_frame_metrics=save_per_frame_metrics)
    metrics_tracker.initialise_metric_sums()
    metrics_tracker.initialise_per_frame_metric_lists()

    if save_per_frame_metrics:
        fname_per_frame = []
        pose_per_frame = []
        shape_per_frame = []
        cam_per_frame = []

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            # ------------------------------- TARGETS and INPUTS -------------------------------
            input = samples_batch['input']
            input = input.to(device)
            target_pose = samples_batch['pose'].to(device)
            target_shape = samples_batch['shape'].to(device)
            target_gender = samples_batch['gender'][0]
            fname = samples_batch['fname']

            edge_detector_output = edge_detector(input[:, :3, :, :])
            rgb_in = edge_detector_output['thresholded_thin_edges'] if edge_nms else edge_detector_output['thresholded_grad_magnitude']
            heatmaps_in = input[:, 3:, :, :]
            input = torch.cat([rgb_in, heatmaps_in], dim=1)

            # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
            target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
            target_glob_rotmats = target_pose_rotmats[:, [0], :, :]
            target_glob_vecs, _ = rotate_global_pose_rotmats_torch(axis=[1, 0, 0],
                                                                   angle=np.pi,
                                                                   glob_rotmats=target_glob_rotmats,
                                                                   rot_mult_order='pre')
            target_pose[:, :3] = target_glob_vecs

            if target_gender == 'm':
                target_smpl_output = smpl_male(body_pose=target_pose[:, 3:],
                                               global_orient=target_pose[:, :3],
                                               betas=target_shape)
                target_reposed_smpl_output = smpl_male(betas=target_shape)
            elif target_gender == 'f':
                target_smpl_output = smpl_female(body_pose=target_pose[:, 3:],
                                                 global_orient=target_pose[:, :3],
                                                 betas=target_shape)
                target_reposed_smpl_output = smpl_female(betas=target_shape)

            target_vertices = target_smpl_output.vertices
            target_reposed_vertices = target_reposed_smpl_output.vertices
            target_joints_all = target_smpl_output.joints
            target_joints_h36mlsp = target_joints_all[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]

            # ------------------------------- PREDICTIONS -------------------------------
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
            pred_shape_dist, pred_glob, pred_cam_wp = model(input)
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

            pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)

            pred_smpl_output_mode = smpl(body_pose=pred_pose_rotmats_mode,
                                         global_orient=pred_glob_rotmats.unsqueeze(1),
                                         betas=pred_shape_dist.loc,
                                         pose2rot=False)
            pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
            pred_joints_all_mode = pred_smpl_output_mode.joints
            pred_joints_h36mlsp_mode = pred_joints_all_mode[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]  # (1, 14, 3)

            pred_reposed_vertices_mean = smpl(betas=pred_shape_dist.loc).vertices  # (1, 6890, 3)

            if any('samples' in metric for metric in metrics_to_track):
                assert pred_pose_F.shape[0] == 1, "Batch size must be 1 for min samples metrics!"
                pred_pose_rotmats_samples = pose_matrix_fisher_sampling_torch(pose_U=pred_pose_U,
                                                                              pose_S=pred_pose_S,
                                                                              pose_V=pred_pose_V,
                                                                              num_samples=num_samples_for_metrics,
                                                                              b=1.5,
                                                                              oversampling_ratio=8,
                                                                              sample_on_cpu=sample_on_cpu)  # (1, num samples, 23, 3, 3)
                pred_shape_samples = pred_shape_dist.rsample([num_samples_for_metrics]).transpose(0, 1)  # (1, num_samples, num_smpl_betas)
                pred_smpl_output_samples = smpl(body_pose=pred_pose_rotmats_samples[0, :, :, :, :],
                                                global_orient=pred_glob_rotmats.unsqueeze(1).expand(num_samples_for_metrics, -1, -1, -1),
                                                betas=pred_shape_samples[0, :, :],
                                                pose2rot=False)
                pred_vertices_samples = pred_smpl_output_samples.vertices
                pred_vertices_samples[0] = pred_vertices_mode[0]  # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics
                pred_joints_h36mlsp_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]
                pred_joints_h36mlsp_samples[0] = pred_joints_h36mlsp_mode[0]  # (num samples, 14, 3) - Including mode as one of the samples for 3D samples min metrics

                pred_reposed_vertices_samples = smpl(body_pose=torch.zeros(num_samples_for_metrics, 69, device=device).float(),
                                                     global_orient=torch.zeros(num_samples_for_metrics, 3, device=device).float(),
                                                     betas=pred_shape_samples[0, :, :]).vertices
                pred_reposed_vertices_samples[0] = pred_reposed_vertices_mean[0]  # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics

            # ------------------------------- TRACKING METRICS -------------------------------
            # Numpy-fying targets
            target_vertices = target_vertices.cpu().detach().numpy()
            target_joints_h36mlsp = target_joints_h36mlsp.cpu().detach().numpy()
            target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

            # Numpy-fying preds
            pred_vertices_mode = pred_vertices_mode.cpu().detach().numpy()
            pred_joints_h36mlsp_mode = pred_joints_h36mlsp_mode.cpu().detach().numpy()
            pred_reposed_vertices_mean = pred_reposed_vertices_mean.cpu().detach().numpy()

            # Update metrics
            pred_dict = {'verts': pred_vertices_mode,
                         'reposed_verts': pred_reposed_vertices_mean,
                         'joints3D': pred_joints_h36mlsp_mode}
            target_dict = {'verts': target_vertices,
                           'reposed_verts': target_reposed_vertices,
                           'joints3D': target_joints_h36mlsp}

            if any('samples_min' in metric for metric in metrics_to_track):
                pred_dict['verts_samples'] = pred_vertices_samples.cpu().detach().numpy()
                pred_dict['reposed_verts_samples'] = pred_reposed_vertices_samples.cpu().detach().numpy()
                pred_dict['joints3D_samples'] = pred_joints_h36mlsp_samples.cpu().detach().numpy()

            transformed_points, per_frame_metrics = metrics_tracker.update_per_batch(pred_dict,
                                                                                     target_dict,
                                                                                     1,
                                                                                     return_transformed_points=visualise,
                                                                                     return_per_frame_metrics=visualise)
            if save_per_frame_metrics:
                fname_per_frame.append(fname)
                pose_per_frame.append(np.concatenate([pred_glob_rotmats[:, None, :, :].cpu().detach().numpy(),
                                                      pred_pose_rotmats_mode.cpu().detach().numpy()],
                                                     axis=1))
                shape_per_frame.append(pred_shape_dist.loc.cpu().detach().numpy())
                cam_per_frame.append(pred_cam_wp.cpu().detach().numpy())

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    metrics_tracker.compute_final_metrics()

    if save_per_frame_metrics:
        fname_per_frame = np.concatenate(fname_per_frame, axis=0)
        np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)

        pose_per_frame = np.concatenate(pose_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

        shape_per_frame = np.concatenate(shape_per_frame, axis=0)
        np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

        cam_per_frame = np.concatenate(cam_per_frame, axis=0)
        np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)


