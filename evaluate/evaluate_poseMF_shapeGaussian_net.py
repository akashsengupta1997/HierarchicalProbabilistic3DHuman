import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from smplx.lbs import batch_rodrigues

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from metrics.eval_metrics_tracker import EvalMetricsTracker

from utils.cam_utils import orthographic_project_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_multiclass_to_binary_labels, ALL_JOINTS_TO_COCO_MAP, ALL_JOINTS_TO_H36M_MAP, H36M_TO_J14
from utils.sampling_utils import pose_matrix_fisher_sampling_torch


def evaluate_pose_MF_shapeGaussian_net(pose_shape_model,
                                       pose_shape_cfg,
                                       smpl_model,
                                       smpl_model_male,
                                       smpl_model_female,
                                       edge_detect_model,
                                       device,
                                       eval_dataset,
                                       metrics,
                                       save_path,
                                       num_workers=4,
                                       pin_memory=True,
                                       save_per_frame_metrics=True,
                                       num_samples_for_metrics=10,
                                       sample_on_cpu=False):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    # Instantiate metrics tracker
    metrics_tracker = EvalMetricsTracker(metrics,
                                         save_path=save_path,
                                         save_per_frame_metrics=save_per_frame_metrics)
    metrics_tracker.initialise_metric_sums()
    metrics_tracker.initialise_per_frame_metric_lists()

    if any('silhouette' in metric for metric in metrics):
        silhouette_renderer = TexturedIUVRenderer(device=device,
                                                  batch_size=1,
                                                  img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                  projection_type='orthographic',
                                                  render_rgb=False,
                                                  bin_size=32)

    if save_per_frame_metrics:
        fname_per_frame = []
        pose_per_frame = []
        shape_per_frame = []
        cam_per_frame = []

    pose_shape_model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            # ------------------ INPUTS ------------------
            image = samples_batch['image'].to(device)
            heatmaps = samples_batch['heatmaps'].to(device)
            edge_detector_output = edge_detect_model(image)
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_input = torch.cat([proxy_rep_img, heatmaps], dim=1)

            # ------------------ Targets ------------------
            target_pose = samples_batch['pose'].to(device)
            target_shape = samples_batch['shape'].to(device)
            target_gender = samples_batch['gender'][0]
            fname = samples_batch['fname']
            if any('joints2D' in metric for metric in metrics):
                target_joints2d_coco = samples_batch['keypoints']
            if any('silhouette' in metric for metric in metrics):
                target_silhouette = samples_batch['silhouette']

            # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
            target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
            target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
            target_glob_vecs, _ = aa_rotate_rotmats(rotmats=target_glob_rotmats,
                                                    axis=[1, 0, 0],
                                                    angle=np.pi,
                                                    rot_mult_order='pre')
            target_pose[:, :3] = target_glob_vecs

            if target_gender == 'm':
                target_smpl_output = smpl_model_male(body_pose=target_pose[:, 3:],
                                                     global_orient=target_pose[:, :3],
                                                     betas=target_shape)
                target_reposed_smpl_output = smpl_model_male(betas=target_shape)
            elif target_gender == 'f':
                target_smpl_output = smpl_model_female(body_pose=target_pose[:, 3:],
                                                       global_orient=target_pose[:, :3],
                                                       betas=target_shape)
                target_reposed_smpl_output = smpl_model_female(betas=target_shape)

            target_vertices = target_smpl_output.vertices
            target_reposed_vertices = target_reposed_smpl_output.vertices
            target_joints_h36mlsp = target_smpl_output.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]

            # ------------------------------- PREDICTIONS -------------------------------
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
            pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

            orthographic_scale = pred_cam_wp[:, [0, 0]]
            cam_t = torch.cat([pred_cam_wp[:, 1:],
                               torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            if pred_glob.shape[-1] == 3:
                pred_glob_rotmats = batch_rodrigues(pred_glob)  # (1, 3, 3)
            elif pred_glob.shape[-1] == 6:
                pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)

            pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                               global_orient=pred_glob_rotmats.unsqueeze(1),
                                               betas=pred_shape_dist.loc,
                                               pose2rot=False)
            pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
            pred_joints_all_mode = pred_smpl_output_mode.joints
            pred_joints_h36mlsp_mode = pred_joints_all_mode[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]  # (1, 14, 3)
            pred_joints_coco_mode = pred_joints_all_mode[:, ALL_JOINTS_TO_COCO_MAP, :]

            pred_reposed_smpl_output_mean = smpl_model(betas=pred_shape_dist.loc)
            pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices  # (1, 6890, 3)

            # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
            # Need to flip pred vertices and pred joints before projecting to 2D for 2D metrics
            if any('joints2D' in metric for metric in metrics):
                pred_joints_coco_mode = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_mode,
                                                                             axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                             angles=np.pi,
                                                                             translations=torch.zeros(3, device=device).float())
                pred_joints2d_coco_mode = orthographic_project_torch(pred_joints_coco_mode, pred_cam_wp)  # (1, 17, 2)
                pred_joints2d_coco_mode = undo_keypoint_normalisation(pred_joints2d_coco_mode, pose_shape_cfg.DATA.PROXY_REP_SIZE)
            if any('silhouette' in metric for metric in metrics):
                pred_vertices_flipped_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                                  axes=torch.tensor([1., 0., 0.], device=device),
                                                                                  angles=np.pi,
                                                                                  translations=torch.zeros(3, device=device))

            if 'silhouette-IOU' in metrics:
                wp_render_output = silhouette_renderer(vertices=pred_vertices_flipped_mode,
                                                       cam_t=cam_t,
                                                       orthographic_scale=orthographic_scale)
                iuv_mode = wp_render_output['iuv_images']
                part_seg_mode = iuv_mode[:, :, :, 0].round()
                pred_silhouette_mode = convert_multiclass_to_binary_labels(part_seg_mode)

            if any('samples' in metric for metric in metrics):
                assert pred_pose_F.shape[0] == 1, "Batch size must be 1 for min samples metrics!"
                pred_pose_rotmats_samples = pose_matrix_fisher_sampling_torch(pose_U=pred_pose_U,
                                                                              pose_S=pred_pose_S,
                                                                              pose_V=pred_pose_V,
                                                                              num_samples=num_samples_for_metrics,
                                                                              b=1.5,
                                                                              oversampling_ratio=8,
                                                                              sample_on_cpu=sample_on_cpu)  # (1, num samples, 23, 3, 3)
                pred_shape_samples = pred_shape_dist.rsample([num_samples_for_metrics]).transpose(0, 1)  # (1, num_samples, num_smpl_betas)
                pred_smpl_output_samples = smpl_model(body_pose=pred_pose_rotmats_samples[0, :, :, :, :],
                                                      global_orient=pred_glob_rotmats.unsqueeze(1).expand(num_samples_for_metrics, -1, -1, -1),
                                                      betas=pred_shape_samples[0, :, :],
                                                      pose2rot=False)
                pred_vertices_samples = pred_smpl_output_samples.vertices
                pred_vertices_samples[0] = pred_vertices_mode[0]  # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics
                pred_joints_h36mlsp_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]
                pred_joints_h36mlsp_samples[0] = pred_joints_h36mlsp_mode[0]  # (num samples, 14, 3) - Including mode as one of the samples for 3D samples min metrics

                pred_reposed_vertices_samples = smpl_model(body_pose=torch.zeros(num_samples_for_metrics, 69, device=device).float(),
                                                           global_orient=torch.zeros(num_samples_for_metrics, 3, device=device).float(),
                                                           betas=pred_shape_samples[0, :, :]).vertices
                pred_reposed_vertices_samples[0] = pred_reposed_vertices_mean[0]   # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics

                if 'joints2Dsamples-L2E' in metrics:
                    pred_joints_coco_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_COCO_MAP, :]  # (num samples, 17, 3)
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip pred_joints_coco 180° about x-axis so they are right way up when projected
                    pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                                    axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                                    angles=np.pi,
                                                                                    translations=torch.zeros(3, device=device).float())
                    pred_joints2d_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)  # (num samples, 17, 2)
                    pred_joints2d_coco_samples = undo_keypoint_normalisation(pred_joints2d_coco_samples, pose_shape_cfg.DATA.PROXY_REP_SIZE)

                if 'silhouettesamples-IOU' in metrics:
                    pred_silhouette_samples = []
                    for i in range(num_samples_for_metrics):
                        pred_vertices_flipped_sample = aa_rotate_translate_points_pytorch3d(points=pred_smpl_output_samples.vertices[[i]],
                                                                                            axes=torch.tensor([1., 0., 0.], device=device),
                                                                                            angles=np.pi,
                                                                                            translations=torch.zeros(3, device=device))
                        iuv_sample = silhouette_renderer(vertices=pred_vertices_flipped_sample,
                                                         cam_t=cam_t,
                                                         orthographic_scale=orthographic_scale)['iuv_images']
                        part_seg_sample = iuv_sample[:, :, :, 0].round()
                        pred_silhouette_samples.append(convert_multiclass_to_binary_labels(part_seg_sample))
                    pred_silhouette_samples = torch.stack(pred_silhouette_samples, dim=1)  # (1, num samples, img wh, img wh)

            # ------------------------------- TRACKING METRICS -------------------------------
            pred_dict = {'verts': pred_vertices_mode.cpu().detach().numpy(),
                         'reposed_verts': pred_reposed_vertices_mean.cpu().detach().numpy(),
                         'joints3D': pred_joints_h36mlsp_mode.cpu().detach().numpy()}
            target_dict = {'verts': target_vertices.cpu().detach().numpy(),
                           'reposed_verts': target_reposed_vertices.cpu().detach().numpy(),
                           'joints3D': target_joints_h36mlsp.cpu().detach().numpy()}

            if 'joints2D-L2E' in metrics:
                pred_dict['joints2D'] = pred_joints2d_coco_mode.cpu().detach().numpy()
                target_dict['joints2D'] = target_joints2d_coco.numpy()
            if 'silhouette-IOU' in metrics:
                pred_dict['silhouettes'] = pred_silhouette_mode.cpu().detach().numpy()
                target_dict['silhouettes'] = target_silhouette.numpy()

            if any('samples_min' in metric for metric in metrics):
                pred_dict['verts_samples'] = pred_vertices_samples.cpu().detach().numpy()
                pred_dict['reposed_verts_samples'] = pred_reposed_vertices_samples.cpu().detach().numpy()
                pred_dict['joints3D_samples'] = pred_joints_h36mlsp_samples.cpu().detach().numpy()
            if 'joints2Dsamples-L2E' in metrics:
                pred_dict['joints2Dsamples'] = pred_joints2d_coco_samples[None, :, :, :].cpu().detach().numpy()
            if 'silhouettesamples-IOU' in metrics:
                pred_dict['silhouettessamples'] = pred_silhouette_samples.cpu().detach().numpy()

            metrics_tracker.update_per_batch(pred_dict,
                                             target_dict,
                                             1,
                                             return_transformed_points=False,
                                             return_per_frame_metrics=False)

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
