import matplotlib
matplotlib.use('agg')
import os
import cv2
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from smplx.lbs import batch_rodrigues
import config

from models.smpl_official import SMPL
from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from metrics.eval_metrics_tracker import EvalMetricsTracker

from utils.cam_utils import orthographic_project_torch
from utils.image_utils import batch_add_rgb_background
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d, rotate_global_pose_rotmats_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_multiclass_to_binary_labels
from utils.sampling_utils import compute_vertex_uncertainties_by_pose_MF_shape_gaussian_sampling, \
    pose_matrix_fisher_sampling_torch, joints2D_error_sorted_verts_sampling


def evaluate_single_in_kinematic_pose_mf_shape_gaussian_with_glob_cam_ssp3d(model,
                                                                            eval_dataset,
                                                                            metrics_to_track,
                                                                            device,
                                                                            save_path,
                                                                            smpl=None,
                                                                            num_workers=4,
                                                                            pin_memory=True,
                                                                            visualise=True,
                                                                            vis_every_n_batches=1000,
                                                                            img_wh=256,
                                                                            save_per_frame_metrics=True,
                                                                            save_per_frame_uncertainty=False,
                                                                            edge_detector=None,
                                                                            edge_nms=False,
                                                                            vis_img_wh=256,
                                                                            num_samples_to_visualise=0,
                                                                            # num_samples_for_joints2Dsamples_l2e=8,
                                                                            num_samples_for_metrics=10,
                                                                            sample_on_cpu=False,
                                                                            j2D_l2e_sort_for_visualise=False,
                                                                            vis_unsorted_samples=True,
                                                                            save_samples=False,
                                                                            occlude=None,
                                                                            num_smpl_betas=10,
                                                                            render_pretty_reposed=False):
    """
    Evaluator for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on SSP3D.
    Input --> ResNet --> image features --> FC layers --> Matrix Fisher over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to Distribution predictor.
    """
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    if smpl is None:
        smpl = SMPL(config.SMPL_300_MODEL_DIR, batch_size=1, num_betas=num_smpl_betas)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female')
    smpl.to(device)
    smpl_male.to(device)
    smpl_female.to(device)

    if 'silhouette_ious' in metrics_to_track:
        wp_renderer = TexturedIUVRenderer(device=device,
                                          batch_size=1,
                                          img_wh=img_wh,
                                          projection_type='orthographic',
                                          render_rgb=False,
                                          bin_size=32)

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
    if save_per_frame_uncertainty:
        vertices_uncertainty_per_frame = []

    # Setting up body visualisation renderer
    body_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=1,
                                            img_wh=vis_img_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            bin_size=32)
    texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
    reposed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
    reposed_orthographic_scale = torch.tensor([[0.85, 0.85]], device=device)

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            # ------------------------------- TARGETS and INPUTS -------------------------------
            input = samples_batch['input']
            input = input.to(device)
            target_pose = samples_batch['pose'].to(device)
            target_shape = samples_batch['shape'].to(device)
            target_gender = samples_batch['gender'][0]
            target_joints2d_coco = samples_batch['keypoints']
            target_silhouette = samples_batch['silhouette']
            fname = samples_batch['fname']
            if visualise and batch_num % vis_every_n_batches == 0:
                vis_mask = samples_batch['vis_mask'].to(device)
                if occlude == 'bottom':
                    vis_mask[:, vis_img_wh // 2:, :] = 0.
                elif occlude == 'side':
                    vis_mask[:, :, vis_img_wh // 2:] = 0.

            if occlude == 'bottom':
                input[:, :, img_wh // 2:, :] = 0.
            elif occlude == 'side':
                input[:, :, :, img_wh // 2:] = 0.

            if edge_detector is not None:
                # Edge detection on input images
                assert input.shape[1] == 3 or input.shape[1] == 3 + 17
                rgb_in = input[:, :3, :, :]
                edge_detector_output = edge_detector(rgb_in)
                rgb_in = edge_detector_output['thresholded_thin_edges'] if edge_nms else edge_detector_output['thresholded_grad_magnitude']
                if input.shape[1] == 3:
                    input = rgb_in
                elif input.shape[1] == 3 + 17:
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
            target_reposed_joints = target_reposed_smpl_output.joints

            # ------------------------------- PREDICTIONS -------------------------------
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
            pred_shape_dist, pred_glob, pred_cam_wp = model(input)
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

            orthographic_scale = pred_cam_wp[:, [0, 0]]
            cam_t = torch.cat([pred_cam_wp[:, 1:],
                               torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            if pred_glob.shape[-1] == 3:
                pred_glob_rotmats = batch_rodrigues(pred_glob)  # (1, 3, 3)
            elif pred_glob.shape[-1] == 6:
                pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)

            pred_smpl_output_mode = smpl(body_pose=pred_pose_rotmats_mode,
                                         global_orient=pred_glob_rotmats.unsqueeze(1),
                                         betas=pred_shape_dist.loc,
                                         pose2rot=False)
            pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
            pred_joints_all_mode = pred_smpl_output_mode.joints
            # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
            # Need to flip pred_vertices before projecting to get pred_vertices_2D
            pred_joints_coco_mode = pred_joints_all_mode[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints_coco_mode = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_mode,
                                                                         axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                         angles=np.pi,
                                                                         translations=torch.zeros(3, device=device).float())
            pred_vertices_flipped_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                              axes=torch.tensor([1., 0., 0.], device=device),
                                                                              angles=np.pi,
                                                                              translations=torch.zeros(3, device=device))
            pred_vertices2d_mode = orthographic_project_torch(pred_vertices_flipped_mode, pred_cam_wp, scale_first=False)
            pred_vertices2d_mode = undo_keypoint_normalisation(pred_vertices2d_mode, img_wh)
            pred_joints2d_coco_mode = orthographic_project_torch(pred_joints_coco_mode, pred_cam_wp)  # (1, 17, 2)
            pred_joints2d_coco_mode = undo_keypoint_normalisation(pred_joints2d_coco_mode, img_wh)

            pred_reposed_smpl_output_mean = smpl(betas=pred_shape_dist.loc)
            pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices  # (1, 6890, 3)
            pred_reposed_vertices_flipped_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_mean,
                                                                                      axes=torch.tensor([1., 0., 0.], device=device),
                                                                                      angles=np.pi,
                                                                                      translations=torch.zeros(3, device=device))
            pred_reposed_joints_mean = pred_reposed_smpl_output_mean.joints

            if 'silhouette_ious' in metrics_to_track:
                wp_render_output = wp_renderer(vertices=pred_vertices_flipped_mode,
                                               cam_t=cam_t,
                                               orthographic_scale=orthographic_scale)
                iuv_mode = wp_render_output['iuv_images']
                part_seg_mode = iuv_mode[:, :, :, 0].round()
                pred_silhouette_mode = convert_multiclass_to_binary_labels(part_seg_mode)

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

                pred_reposed_vertices_samples = smpl(body_pose=torch.zeros(num_samples_for_metrics, 69, device=device).float(),
                                                     global_orient=torch.zeros(num_samples_for_metrics, 3, device=device).float(),
                                                     betas=pred_shape_samples[0, :, :]).vertices
                pred_reposed_vertices_samples[0] = pred_reposed_vertices_mean[0]   # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics

                if 'joints2Dsamples_l2es' in metrics_to_track:
                    pred_joints_coco_samples = pred_smpl_output_samples.joints[:, config.ALL_JOINTS_TO_COCO_MAP, :]  # (num samples, 17, 3)
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip pred_joints_coco 180° about x-axis so they are right way up when projected
                    pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                                    axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                                    angles=np.pi,
                                                                                    translations=torch.zeros(3, device=device).float())
                    pred_joints2d_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)  # (num samples, 17, 2)
                    pred_joints2d_coco_samples = undo_keypoint_normalisation(pred_joints2d_coco_samples, img_wh)

                if 'silhouettesamples_ious' in metrics_to_track:
                    pred_silhouette_samples = []
                    for i in range(num_samples_for_metrics):
                        pred_vertices_flipped_sample = aa_rotate_translate_points_pytorch3d(points=pred_smpl_output_samples.vertices[[i]],
                                                                                            axes=torch.tensor([1., 0., 0.], device=device),
                                                                                            angles=np.pi,
                                                                                            translations=torch.zeros(3, device=device))
                        iuv_sample = wp_renderer(vertices=pred_vertices_flipped_sample,
                                                 cam_t=cam_t,
                                                 orthographic_scale=orthographic_scale)['iuv_images']
                        part_seg_sample = iuv_sample[:, :, :, 0].round()
                        pred_silhouette_samples.append(convert_multiclass_to_binary_labels(part_seg_sample))
                    pred_silhouette_samples = torch.stack(pred_silhouette_samples, dim=1)  # (1, num samples, img wh, img wh)

            # ------------------------------- TRACKING METRICS -------------------------------
            # Numpy-fying targets
            target_vertices = target_vertices.cpu().detach().numpy()
            target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
            target_reposed_joints = target_reposed_joints.cpu().detach().numpy()
            target_silhouette = target_silhouette.numpy()
            target_joints2d_coco = target_joints2d_coco.numpy()
            target_pose = target_pose.cpu().detach().numpy()
            target_shape = target_shape.cpu().detach().numpy()

            # Numpy-fying preds
            pred_vertices_mode = pred_vertices_mode.cpu().detach().numpy()
            pred_reposed_vertices_mean = pred_reposed_vertices_mean.cpu().detach().numpy()
            pred_reposed_joints_mean = pred_reposed_joints_mean.cpu().detach().numpy()
            pred_vertices2d_mode = pred_vertices2d_mode.cpu().detach().numpy()
            pred_joints2d_coco_mode = pred_joints2d_coco_mode.cpu().detach().numpy()
            if 'silhouette_ious' in metrics_to_track:
                pred_silhouette_mode = pred_silhouette_mode.cpu().detach().numpy()

            pred_dict = {'verts': pred_vertices_mode,
                         'reposed_verts': pred_reposed_vertices_mean,
                         'joints2D': pred_joints2d_coco_mode,
                         'reposed_joints': pred_reposed_joints_mean}
            target_dict = {'verts': target_vertices,
                           'reposed_verts': target_reposed_vertices,
                           'joints2D': target_joints2d_coco,
                           'reposed_joints': target_reposed_joints}
            if 'silhouette_ious' in metrics_to_track:
                pred_dict['silhouettes'] = pred_silhouette_mode
                target_dict['silhouettes'] = target_silhouette
            if 'joints2Dsamples_l2es' in metrics_to_track:
                pred_dict['joints2Dsamples'] = pred_joints2d_coco_samples[None, :, :, :].cpu().detach().numpy()
            if 'silhouettesamples_ious' in metrics_to_track:
                pred_dict['silhouettessamples'] = pred_silhouette_samples.cpu().detach().numpy()
            if any('samples_min' in metric for metric in metrics_to_track):
                pred_dict['verts_samples'] = pred_vertices_samples.cpu().detach().numpy()
                pred_dict['reposed_verts_samples'] = pred_reposed_vertices_samples.cpu().detach().numpy()

            num_samples_in_batch = target_shape.shape[0]
            transformed_points, per_frame_metrics = metrics_tracker.update_per_batch(pred_dict, target_dict,
                                                                                     num_samples_in_batch,
                                                                                     return_transformed_points=visualise,
                                                                                     return_per_frame_metrics=visualise)

            if save_per_frame_metrics:
                fname_per_frame.append(fname)
                pose_per_frame.append(np.concatenate([pred_glob_rotmats[:, None, :, :].cpu().detach().numpy(),
                                                      pred_pose_rotmats_mode.cpu().detach().numpy()],
                                                     axis=1))
                shape_per_frame.append(pred_shape_dist.loc.cpu().detach().numpy())
                cam_per_frame.append(pred_cam_wp.cpu().detach().numpy())

            # ------------------------------- VISUALISE -------------------------------
            if visualise and batch_num % vis_every_n_batches == 0:
                # Body visualisation
                body_vis_output = body_vis_renderer(vertices=pred_vertices_flipped_mode[[0]],
                                                    textures=texture,
                                                    cam_t=cam_t,
                                                    orthographic_scale=orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings)
                body_vis_rgb = body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous()
                body_vis_seg = body_vis_output['iuv_images'][:, :, :, 0].round()
                body_vis_rgb = batch_add_rgb_background(backgrounds=vis_mask.permute(0, 3, 1, 2).contiguous(),
                                                        rgb=body_vis_rgb,
                                                        seg=body_vis_seg).cpu().detach().numpy()[0].transpose(1, 2, 0)
                pred_vertices_vis_rot = aa_rotate_translate_points_pytorch3d(points=pred_vertices_flipped_mode[[0]],
                                                                             axes=torch.tensor([0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))
                body_vis_rgb_rot = body_vis_renderer(vertices=pred_vertices_vis_rot,
                                                     textures=texture,
                                                     cam_t=cam_t,
                                                     orthographic_scale=orthographic_scale,
                                                     lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                # Reposed body visualisation
                reposed_body_vis_rgb = body_vis_renderer(vertices=pred_reposed_vertices_flipped_mean[[0]],
                                                         textures=texture,
                                                         cam_t=reposed_cam_t,
                                                         orthographic_scale=reposed_orthographic_scale,
                                                         lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                pred_reposed_vertices_vis_rot = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_flipped_mean[[0]],
                                                                                     axes=torch.tensor([0., 1., 0.], device=device),
                                                                                     angles=-np.pi / 2.,
                                                                                     translations=torch.zeros(3, device=device))
                reposed_body_vis_rgb_rot = body_vis_renderer(vertices=pred_reposed_vertices_vis_rot,
                                                             textures=texture,
                                                             cam_t=reposed_cam_t,
                                                             orthographic_scale=reposed_orthographic_scale,
                                                             lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

                # Uncertainty Computation
                # Uncertainty computed by sampling + average distance from mean
                avg_vertices_distance_from_mean, avg_vertices_sc_distance_from_mean, pred_vertices_samples_for_vis, pred_joints_samples_for_vis = \
                    compute_vertex_uncertainties_by_pose_MF_shape_gaussian_sampling(pose_U=pred_pose_U,
                                                                                    pose_S=pred_pose_S,
                                                                                    pose_V=pred_pose_V,
                                                                                    shape_distribution=pred_shape_dist,
                                                                                    glob_rotmats=pred_glob_rotmats,
                                                                                    num_samples=60,
                                                                                    smpl_model=smpl,
                                                                                    reposed=False,
                                                                                    target_vertices=target_vertices)
                if save_per_frame_uncertainty:
                    vertices_uncertainty_per_frame.append(avg_vertices_distance_from_mean)
                avg_reposed_vertices_distance_from_mean, _, _, avg_reposed_vertices_distance_from_mean_xyz, _ = \
                    compute_vertex_uncertainties_by_pose_MF_shape_gaussian_sampling(pose_U=None,
                                                                                    pose_S=None,
                                                                                    pose_V=None,
                                                                                    shape_distribution=pred_shape_dist,
                                                                                    glob_rotmats=pred_glob_rotmats,
                                                                                    num_samples=60,
                                                                                    smpl_model=smpl,
                                                                                    reposed=True,
                                                                                    target_vertices=target_reposed_vertices,
                                                                                    return_separate_reposed_dims=True)

                if num_samples_to_visualise > 0:
                    # Samples for vis using mean shape, useful for checking whether pose distribution matches image.
                    _, _, pred_vertices_samples_for_vis_mean_shape, pred_joints_samples_for_vis_mean_shape = \
                        compute_vertex_uncertainties_by_pose_MF_shape_gaussian_sampling(pose_U=pred_pose_U,
                                                                                        pose_S=pred_pose_S,
                                                                                        pose_V=pred_pose_V,
                                                                                        shape_distribution=pred_shape_dist,
                                                                                        glob_rotmats=pred_glob_rotmats,
                                                                                        num_samples=6*num_samples_to_visualise,
                                                                                        smpl_model=smpl,
                                                                                        reposed=False,
                                                                                        target_vertices=target_vertices,
                                                                                        use_mean_shape=True)
                    if j2D_l2e_sort_for_visualise:
                        # Samples for vis using 2D Joint Error-based Rejection Sampling (i.e. input consistent samples)
                        pred_vertices_j2d_l2e_sort_samples = joints2D_error_sorted_verts_sampling(
                            pred_vertices_samples=torch.from_numpy(pred_vertices_samples_for_vis[:3*num_samples_to_visualise, :, :]).float().to(device),
                            pred_joints_samples=torch.from_numpy(pred_joints_samples_for_vis[:3*num_samples_to_visualise, :, :]).float().to(device),
                            input_joints2D_heatmaps=input[:, 1:, :, :],
                            pred_cam_wp=pred_cam_wp)[:num_samples_to_visualise, :, :]  # (num samples to vis, 6890, 3)
                        pred_vertices_j2d_l2e_sort_samples_mean_shape = joints2D_error_sorted_verts_sampling(
                            pred_vertices_samples=torch.from_numpy(pred_vertices_samples_for_vis_mean_shape[:3 * num_samples_to_visualise, :, :]).float().to(device),
                            pred_joints_samples=torch.from_numpy(pred_joints_samples_for_vis_mean_shape[:3 * num_samples_to_visualise, :, :]).float().to(device),
                            input_joints2D_heatmaps=input[:, 1:, :, :],
                            pred_cam_wp=pred_cam_wp)[:num_samples_to_visualise, :, :]  # (num samples to vis, 6890, 3)

                        pred_vertices_j2d_l2e_sort_samples = torch.cat([pred_vertices_j2d_l2e_sort_samples,
                                                                        pred_vertices_j2d_l2e_sort_samples_mean_shape],
                                                                       dim=0)  # (2 * num samples to vis, 6890, 3)
                        pred_vertices_j2d_l2e_sort_samples = aa_rotate_translate_points_pytorch3d(
                            points=pred_vertices_j2d_l2e_sort_samples,
                            axes=torch.tensor([1., 0., 0.], device=device),
                            angles=np.pi,
                            translations=torch.zeros(3, device=device))
                        pred_vertices_j2d_l2e_sort_samples_rot = aa_rotate_translate_points_pytorch3d(
                            points=pred_vertices_j2d_l2e_sort_samples,
                            axes=torch.tensor([0., 1., 0.], device=device),
                            angles=-np.pi / 2.,
                            translations=torch.zeros(3, device=device))

                    pred_vertices_samples_for_vis = torch.cat([torch.from_numpy(pred_vertices_samples_for_vis[:num_samples_to_visualise, :, :]).float().to(device),
                                                               torch.from_numpy(pred_vertices_samples_for_vis_mean_shape[:num_samples_to_visualise, :, :]).float().to(device)],
                                                              dim=0)
                    pred_vertices_samples_for_vis = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples_for_vis,
                                                                                         axes=torch.tensor([1., 0., 0.], device=device),
                                                                                         angles=np.pi,
                                                                                         translations=torch.zeros(3, device=device))
                    pred_vertices_samples_for_vis_rot = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples_for_vis,
                                                                                             axes=torch.tensor([0., 1., 0.], device=device),
                                                                                             angles=-np.pi / 2.,
                                                                                             translations=torch.zeros(3, device=device))
                    body_vis_rgb_samples = []
                    body_vis_rgb_rot_samples = []
                    if j2D_l2e_sort_for_visualise:
                        body_vis_rgb_j2d_l2e_sort_samples = []
                        body_vis_rgb_rot_j2d_l2e_sort_samples = []
                    for i in range(2 * num_samples_to_visualise):
                        body_vis_output_sample = body_vis_renderer(vertices=pred_vertices_samples_for_vis[[i]],
                                                                   textures=texture,
                                                                   cam_t=cam_t,
                                                                   orthographic_scale=orthographic_scale,
                                                                   lights_rgb_settings=lights_rgb_settings)
                        body_vis_rgb_sample = body_vis_output_sample['rgb_images'].permute(0, 3, 1, 2).contiguous()
                        body_vis_seg_sample = body_vis_output_sample['iuv_images'][:, :, :, 0].round()
                        body_vis_rgb_sample = batch_add_rgb_background(backgrounds=vis_mask.permute(0, 3, 1, 2).contiguous(),
                                                                       rgb=body_vis_rgb_sample,
                                                                       seg=body_vis_seg_sample).cpu().detach().numpy()[0].transpose(1, 2, 0)
                        body_vis_rgb_samples.append(body_vis_rgb_sample)

                        body_vis_rgb_rot_sample = body_vis_renderer(vertices=pred_vertices_samples_for_vis_rot[[i]],
                                                                    textures=texture,
                                                                    cam_t=cam_t,
                                                                    orthographic_scale=orthographic_scale,
                                                                    lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                        body_vis_rgb_rot_samples.append(body_vis_rgb_rot_sample)

                        if j2D_l2e_sort_for_visualise:
                            body_vis_output_j2d_l2e_sort_sample = body_vis_renderer(vertices=pred_vertices_j2d_l2e_sort_samples[[i]],
                                                                                    textures=texture,
                                                                                    cam_t=cam_t,
                                                                                    orthographic_scale=orthographic_scale,
                                                                                    lights_rgb_settings=lights_rgb_settings)
                            body_vis_rgb_j2d_l2e_sort_sample = body_vis_output_j2d_l2e_sort_sample['rgb_images'].permute(0, 3, 1, 2).contiguous()
                            body_vis_seg_j2d_l2e_sort_sample = body_vis_output_j2d_l2e_sort_sample['iuv_images'][:, :, :, 0].round()
                            body_vis_rgb_j2d_l2e_sort_sample = batch_add_rgb_background(backgrounds=vis_mask.permute(0, 3, 1, 2).contiguous(),
                                                                                        rgb=body_vis_rgb_j2d_l2e_sort_sample,
                                                                                        seg=body_vis_seg_j2d_l2e_sort_sample).cpu().detach().numpy()[0].transpose(1, 2, 0)
                            body_vis_rgb_j2d_l2e_sort_samples.append(body_vis_rgb_j2d_l2e_sort_sample)

                            body_vis_rgb_rot_j2d_l2e_sort_sample = body_vis_renderer(vertices=pred_vertices_j2d_l2e_sort_samples_rot[[i]],
                                                                                     textures=texture,
                                                                                     cam_t=cam_t,
                                                                                     orthographic_scale=orthographic_scale,
                                                                                     lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                            body_vis_rgb_rot_j2d_l2e_sort_samples.append(body_vis_rgb_rot_j2d_l2e_sort_sample)

                    if save_samples:
                        if j2D_l2e_sort_for_visualise:
                            samples_save_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_j2d_l2e_sort_samples.npy')
                            np.save(samples_save_path, pred_vertices_j2d_l2e_sort_samples.cpu().detach().numpy())
                        else:
                            samples_save_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_samples.npy')
                            np.save(samples_save_path, pred_vertices_samples_for_vis.cpu().detach().numpy())

                input = input.cpu().detach().numpy()
                vis_mask = vis_mask.cpu().detach().numpy()

                if input.shape[1] > 3:  # Input includes 17 joint heatmaps
                    heatmaps_to_plot = np.sum(input[:, -17:, :, :], axis=1)
                    images_to_plot = np.transpose(input[:, :-17, :, :], [0, 2, 3, 1])
                    images_to_plot = images_to_plot + heatmaps_to_plot[:, :, :, None]
                    if images_to_plot.shape[-1] == 3:
                        images_to_plot = np.clip(images_to_plot, a_min=0.0, a_max=1.0)
                else:
                    images_to_plot = np.transpose(input, [0, 2, 3, 1])
                    if images_to_plot.shape[-1] == 1:
                        images_to_plot = images_to_plot.squeeze(axis=-1)

                # ------------------ Model Prediction, Error and Uncertainty Figure ------------------
                num_row = 6
                num_col = 6
                subplot_count = 1
                plt.figure(figsize=(20, 20))

                # Plot image and mask vis
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(vis_mask[0])
                subplot_count += 1

                # Plot pred vertices 2D and body render overlaid over input
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(images_to_plot[0])
                plt.scatter(pred_vertices2d_mode[0, :, 0],
                            pred_vertices2d_mode[0, :, 1],
                            c='r', s=0.01)
                subplot_count += 1

                # Plot body render overlaid on vis image
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(body_vis_rgb)
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(body_vis_rgb_rot)
                subplot_count += 1

                # Plot input
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(images_to_plot[0])
                subplot_count += 1

                # Plot silhouette+J2D and reposed body render
                if 'silhouette_ious' in metrics_to_track:
                    plt.subplot(num_row, num_col, subplot_count)
                    plt.gca().axis('off')
                    plt.imshow(pred_silhouette_mode[0].astype(np.int16) - target_silhouette[0].astype(np.int16))
                    plt.text(10, 10, s='mIOU: {:.4f}'.format(per_frame_metrics['silhouette_ious'][0]))
                if 'joints2D_l2es' in metrics_to_track:
                    for j in range(target_joints2d_coco.shape[1]):
                        plt.scatter(target_joints2d_coco[0, j, 0],
                                    target_joints2d_coco[0, j, 1],
                                    c='b', s=10.0)
                        plt.text(target_joints2d_coco[0, j, 0], target_joints2d_coco[0, j, 1],
                                 str(j))
                        plt.scatter(pred_joints2d_coco_mode[0, j, 0],
                                    pred_joints2d_coco_mode[0, j, 1],
                                    c='r', s=10.0)
                        plt.text(pred_joints2d_coco_mode[0, j, 0],
                                 pred_joints2d_coco_mode[0, j, 1],
                                 str(j))
                    plt.text(10, 30, s='J2D L2E: {:.4f}'.format(per_frame_metrics['joints2D_l2es'][0]))
                subplot_count += 1

                # Plot reposed body render
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(reposed_body_vis_rgb)
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(reposed_body_vis_rgb_rot)
                subplot_count += 1

                # Plot PVE-SC pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(transformed_points['pred_vertices_sc'][0, :, 0],
                            transformed_points['pred_vertices_sc'][0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.scatter(transformed_points['pred_vertices_sc'][0, 2500, 0],
                            transformed_points['pred_vertices_sc'][0, 2500, 1],
                            s=10.,
                            c='black')
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                pve = np.linalg.norm(
                    transformed_points['pred_vertices_sc'][0] - target_vertices[0], axis=-1)
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_sc'][0, :, 0],
                            transformed_points['pred_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=pve,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                pve = np.linalg.norm(
                    transformed_points['pred_vertices_sc'][0] - target_vertices[0], axis=-1)
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_sc'][0, :, 2],
                            # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=pve,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][0]))
                subplot_count += 1

                # Plot PVE-PA pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-PA')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(transformed_points['pred_vertices_pa'][0, :, 0],
                            transformed_points['pred_vertices_pa'][0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.scatter(transformed_points['pred_vertices_pa'][0, 2500, 0],
                            transformed_points['pred_vertices_pa'][0, 2500, 1],
                            s=10.,
                            c='black')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                pve = np.linalg.norm(
                    transformed_points['pred_vertices_pa'][0] - target_vertices[0], axis=-1)
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_pa'][0, :, 0],
                            transformed_points['pred_vertices_pa'][0, :, 1],
                            s=0.05,
                            c=pve,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                pve = np.linalg.norm(
                    transformed_points['pred_vertices_pa'][0] - target_vertices[0], axis=-1)
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_pa'][0, :, 2],
                            # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_vertices_pa'][0, :, 1],
                            s=0.05,
                            c=pve,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][0]))
                subplot_count += 1

                # Plot PVE-T-SC pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-T-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_reposed_vertices[0, :, 0],
                            target_reposed_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, 2500, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, 2500, 1],
                            s=10.,
                            c='black')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                pve = np.linalg.norm(
                    transformed_points['pred_reposed_vertices_sc'][0] - target_reposed_vertices[
                        0], axis=-1)
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=pve,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                pve = np.linalg.norm(
                    transformed_points['pred_reposed_vertices_sc'][0] - target_reposed_vertices[
                        0], axis=-1)
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 2],  # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=pve,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][0]))
                subplot_count += 1

                # Plot per-vertex uncertainties
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='Uncertainty for\nPVE')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_sc'][0, :, 0],
                            transformed_points['pred_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_vertices_distance_from_mean,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_sc'][0, :, 2],
                            transformed_points['pred_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_vertices_sc_distance_from_mean,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='Uncertainty for\nPVE-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_pa'][0, :, 0],
                            transformed_points['pred_vertices_pa'][0, :, 1],
                            s=0.05,
                            c=avg_vertices_sc_distance_from_mean,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(transformed_points['pred_vertices_pa'][0, :, 2],
                            # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_vertices_pa'][0, :, 1],
                            s=0.05,
                            c=avg_vertices_sc_distance_from_mean,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                # Plot per-reposed-vertex uncertainties in x/y/z directions and sum
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(-0., -0., s='Uncertainty\nfor PVE-T')
                subplot_count += 1

                # x-direction
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.02, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean_xyz[:, 0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.colorbar(shrink=0.5, label='Uncertainty x (m)', orientation='vertical', format='%.2f')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.02, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 2],  # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean_xyz[:, 0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                # y-direction
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.02, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean_xyz[:, 1],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.colorbar(shrink=0.5, label='Uncertainty y (m)', orientation='vertical', format='%.2f')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.02, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 2],  # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean_xyz[:, 1],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                # z-direction
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.02, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean_xyz[:, 2],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.colorbar(shrink=0.5, label='Uncertainty z (m)', orientation='vertical', format='%.2f')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.02, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 2],  # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean_xyz[:, 2],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                # all directions
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.04, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 0],
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.colorbar(shrink=0.5, label='Uncertainty (m)', orientation='vertical', format='%.2f')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.04, clip=True)
                plt.scatter(transformed_points['pred_reposed_vertices_sc'][0, :, 2],  # Equivalent to Rotated 90° about y axis
                            transformed_points['pred_reposed_vertices_sc'][0, :, 1],
                            s=0.05,
                            c=avg_reposed_vertices_distance_from_mean,
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                save_fig_path = os.path.join(save_path, fname[0])
                plt.savefig(save_fig_path, bbox_inches='tight')
                plt.close()

                if render_pretty_reposed:
                    # ---------------------- PRETTY SHAPE UNCERTAINTY RENDER ----------------------
                    all_body_vis_uncertainty_rgb = []
                    var_directions = [0, 2]
                    norm_var = plt.Normalize(vmin=0.0, vmax=0.025, clip=True)
                    for direction in var_directions:
                        for view in ['front', 'side']:
                            var_in_direction_vertices = avg_reposed_vertices_distance_from_mean_xyz[:, direction]
                            vertex_colours = plt.cm.jet(norm_var(var_in_direction_vertices))[:, :3]
                            vertex_colours = torch.from_numpy(vertex_colours[None]).to(device).float()
                            verts_to_render = pred_reposed_vertices_flipped_mean[[0]]
                            if view == 'side':
                                verts_to_render = aa_rotate_translate_points_pytorch3d(points=verts_to_render,
                                                                                       axes=torch.tensor([0., 1., 0.], device=device),
                                                                                       angles=np.pi / 2,
                                                                                       translations=torch.zeros(3, device=device))
                            body_vis_uncertainty_rgb = body_vis_renderer(vertices=verts_to_render,
                                                                         textures=texture,
                                                                         cam_t=reposed_cam_t,
                                                                         orthographic_scale=reposed_orthographic_scale,
                                                                         lights_rgb_settings=lights_rgb_settings,
                                                                         verts_features=vertex_colours)['rgb_images'].cpu().detach().numpy()[0]
                            if view == 'side':
                                body_vis_uncertainty_rgb = body_vis_uncertainty_rgb[:, int(vis_img_wh * 0.375):int(vis_img_wh * 0.625)]
                            all_body_vis_uncertainty_rgb.append(body_vis_uncertainty_rgb)

                    all_body_vis_uncertainty_rgb = np.concatenate(all_body_vis_uncertainty_rgb, axis=1)
                    save_rend_path = os.path.join(save_path, fname[0] + '_uncertainty_render.png')
                    cv2.imwrite(save_rend_path, all_body_vis_uncertainty_rgb[:, :, ::-1] * 255)

                # ------------------ Samples from Predicted Distribution Figure ------------------
                if num_samples_to_visualise > 0 and vis_unsorted_samples:
                    num_subplots = num_samples_to_visualise * 4 + 2
                    num_row = 5
                    num_col = math.ceil(num_subplots / float(num_row))
                    if num_col % 2 == 1:
                        num_col += 1

                    subplot_count = 1
                    plt.figure(figsize=(24, 12))

                    # Plot mode prediction
                    plt.subplot(num_row, num_col, subplot_count)
                    plt.gca().axis('off')
                    plt.imshow(body_vis_rgb)
                    subplot_count += 1

                    plt.subplot(num_row, num_col, subplot_count)
                    plt.gca().axis('off')
                    plt.imshow(body_vis_rgb_rot)
                    subplot_count += 1

                    # Plot samples from predicted distribution
                    for i in range(2 * num_samples_to_visualise):
                        if i == num_samples_to_visualise:
                            subplot_count += 2
                        plt.subplot(num_row, num_col, subplot_count)
                        plt.gca().axis('off')
                        plt.imshow(body_vis_rgb_samples[i])
                        subplot_count += 1

                        plt.subplot(num_row, num_col, subplot_count)
                        plt.gca().axis('off')
                        plt.imshow(body_vis_rgb_rot_samples[i])
                        subplot_count += 1

                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())

                    save_fig_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_samples.png')
                    plt.savefig(save_fig_path, bbox_inches='tight')
                    plt.close()

                # ------------------ J2D Error Sorted Samples from Predicted Distribution Figure ------------------
                if num_samples_to_visualise > 0 and j2D_l2e_sort_for_visualise:
                    num_subplots = num_samples_to_visualise * 4 + 2
                    num_row = 5
                    num_col = math.ceil(num_subplots / float(num_row))
                    if num_col % 2 == 1:
                        num_col += 1

                    subplot_count = 1
                    plt.figure(figsize=(24, 12))

                    # Plot mode prediction
                    plt.subplot(num_row, num_col, subplot_count)
                    plt.gca().axis('off')
                    plt.imshow(body_vis_rgb)
                    subplot_count += 1

                    plt.subplot(num_row, num_col, subplot_count)
                    plt.gca().axis('off')
                    plt.imshow(body_vis_rgb_rot)
                    subplot_count += 1

                    # Plot samples from predicted distribution
                    for i in range(2 * num_samples_to_visualise):
                        if i == num_samples_to_visualise:
                            subplot_count += 2
                        plt.subplot(num_row, num_col, subplot_count)
                        plt.gca().axis('off')
                        plt.imshow(body_vis_rgb_j2d_l2e_sort_samples[i])
                        subplot_count += 1

                        plt.subplot(num_row, num_col, subplot_count)
                        plt.gca().axis('off')
                        plt.imshow(body_vis_rgb_rot_j2d_l2e_sort_samples[i])
                        subplot_count += 1

                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())

                    save_fig_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_j2d_l2e_sort_samples.png')
                    plt.savefig(save_fig_path, bbox_inches='tight')
                    plt.close()

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    metrics_tracker.compute_final_metrics()

    if save_per_frame_metrics:
        fname_per_frame = np.concatenate(fname_per_frame, axis=0)
        np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)
        print(fname_per_frame.shape)

        pose_per_frame = np.concatenate(pose_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)
        print(pose_per_frame.shape)

        shape_per_frame = np.concatenate(shape_per_frame, axis=0)
        np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)
        print(shape_per_frame.shape)

        cam_per_frame = np.concatenate(cam_per_frame, axis=0)
        np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)
        print(cam_per_frame.shape)

        if save_per_frame_uncertainty and visualise:
            vertices_uncertainty_per_frame = np.stack(vertices_uncertainty_per_frame, axis=0)
            np.save(os.path.join(save_path, 'vertices_uncertainty_per_frame.npy'), vertices_uncertainty_per_frame)
            print(vertices_uncertainty_per_frame.shape)