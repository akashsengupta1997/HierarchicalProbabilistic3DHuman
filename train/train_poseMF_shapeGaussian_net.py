import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker

from utils.checkpoint_utils import load_training_info_from_checkpoint
from utils.cam_utils import perspective_project_torch, orthographic_project_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats_pytorch3d
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch, convert_densepose_seg_to_14part_labels
from utils.joints2d_utils import check_joints2d_visibility_torch, check_joints2d_occluded_torch
from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine
from utils.sampling_utils import pose_matrix_fisher_sampling_torch

from utils.augmentation.smpl_augmentation import normal_sample_shape
from utils.augmentation.cam_augmentation import augment_cam_t
from utils.augmentation.proxy_rep_augmentation import augment_proxy_representation, random_extreme_crop
from utils.augmentation.rgb_augmentation import augment_rgb
from utils.augmentation.lighting_augmentation import augment_light

import config


def train_poseMF_shapeGaussian_net(device,
                                   model,
                                   smpl_model,
                                   pytorch3d_renderer,
                                   train_dataset,
                                   val_dataset,
                                   criterion,
                                   optimiser,
                                   batch_size,
                                   num_epochs,
                                   joints2D_loss_on,
                                   num_samples_j2Dloss,
                                   img_wh,
                                   focal_length,
                                   smpl_augment_params,
                                   cam_augment_params,
                                   bbox_augment_params,
                                   proxy_rep_augment_params,
                                   mean_cam_t,
                                   model_save_path,
                                   log_path,
                                   metrics_to_track,
                                   save_val_metrics,
                                   lighting_augment_params=None,
                                   rgb_augment_params=None,
                                   checkpoint=None,
                                   num_workers=0,
                                   pin_memory=False,
                                   epochs_per_save=10,
                                   hmap_gaussian_std=4,
                                   occlude_joints=False,
                                   edge_detector=None,
                                   edge_nms=False,
                                   sample_on_cpu=False,
                                   num_smpl_betas=10,
                                   change_loss_stage_from_epoch=999,
                                   staged_loss_weights=None):
    """
    Input --> ResNet --> image features --> FC layers --> Hierarchical Kinematic Matrix Fisher over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to Gaussian distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    assert joints2D_loss_on in ['means', 'samples', 'means+samples'], "Invalid setting for joints2D_loss_on."

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers,
                                  pin_memory=pin_memory)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                drop_last=True, num_workers=num_workers,
                                pin_memory=pin_memory)
    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}

    # Ensure that all metrics used as model save conditions are being tracked (i.e. that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if checkpoint is not None:
        # Resuming training - note that current model and optimiser state dicts are loaded out
        # of traim function (should be in run file).
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = True
    else:
        current_epoch = 0
        best_epoch_val_metrics = {}
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        load_logs = False

    # Instantiate metrics tracker
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=[],
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=img_wh,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)

    x_axis = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)

    # Starting training loop
    for epoch in range(current_epoch, num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        metrics_tracker.initialise_loss_metric_sums()

        if epoch == change_loss_stage_from_epoch:
            # Apply 3D losses from this epoch onwards.
            criterion.extra_losses_on = ['verts', 'joints3D']
            criterion.loss_weights = staged_loss_weights
            criterion.verts_loss = torch.nn.MSELoss(reduction='mean')
            criterion.joints3D_loss = torch.nn.MSELoss(reduction='mean')
            print('Applying losses on', criterion.extra_losses_on, 'from epoch ', change_loss_stage_from_epoch)
            print('Loss weights:', criterion.loss_weights)
            joints2D_loss_on = 'means+samples'
            print('Applying J2D loss using', joints2D_loss_on, 'with {} samples.'.format(num_samples_j2Dloss))
            print('Sample on CPU:', sample_on_cpu)
            metrics_tracker.metrics_to_track.append('joints2Dsamples L2E')
            print('Tracking metrics:', metrics_tracker.metrics_to_track)

        for split in ['train', 'val']:
            if split == 'train':
                print('Training.')
                model.train()
            else:
                print('Validation.')
                model.eval()

            for batch_num, samples_batch in enumerate(tqdm(dataloaders[split])):
                #############################################################
                # ---------------- SYNTHETIC DATA GENERATION ----------------
                #############################################################
                with torch.no_grad():
                    # ------------ RANDOM POSE, SHAPE, BACKGROUND, TEXTURE, CAMERA SAMPLING ------------
                    # Load target pose and random background/texture
                    target_pose = samples_batch['pose'].to(device)  # (bs, 72)
                    background = samples_batch['background'].to(device)  # (bs, 3, img_wh, img_wh)
                    texture = samples_batch['texture'].to(device)  # (bs, 1200, 800, 3)

                    # Convert target_pose from axis angle to rotmats
                    target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
                    target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
                    target_pose_rotmats = target_pose_rotmats[:, 1:, :, :]
                    # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Then pose predictions will also be right way up in 3D space - network doesn't need to learn to flip.
                    _, target_glob_rotmats = aa_rotate_rotmats_pytorch3d(rotmats=target_glob_rotmats,
                                                                         angles=np.pi,
                                                                         axes=x_axis,
                                                                         rot_mult_order='post')
                    # Random sample body shape
                    target_shape = normal_sample_shape(batch_size=batch_size,
                                                       mean_shape=torch.zeros(num_smpl_betas, device=device).float(),
                                                       std_vector=smpl_augment_params['delta_betas_std_vector'])
                    # Random sample camera translation
                    target_cam_t = augment_cam_t(mean_cam_t,
                                                 xy_std=cam_augment_params['xy_std'],
                                                 delta_z_range=cam_augment_params['delta_z_range'])

                    # Compute target vertices and joints
                    target_smpl_output = smpl_model(body_pose=target_pose_rotmats,
                                                    global_orient=target_glob_rotmats.unsqueeze(1),
                                                    betas=target_shape,
                                                    pose2rot=False)
                    target_vertices = target_smpl_output.vertices
                    target_joints_all = target_smpl_output.joints
                    target_joints_h36m = target_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
                    target_joints_h36mlsp = target_joints_h36m[:, config.H36M_TO_J14, :]

                    target_reposed_vertices = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                         global_orient=torch.zeros_like(target_pose)[:, :3],
                                                         betas=target_shape).vertices

                    # ------------ INPUT PROXY REPRESENTATION GENERATION + 2D TARGET JOINTS ------------
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip target_vertices_for_rendering 180° about x-axis so they are right way up when projected
                    # Need to flip target_joints_coco 180° about x-axis so they are right way up when projected
                    target_vertices_for_rendering = aa_rotate_translate_points_pytorch3d(points=target_vertices,
                                                                                         axes=x_axis,
                                                                                         angles=np.pi,
                                                                                         translations=torch.zeros(3, device=device).float())
                    target_joints_coco = aa_rotate_translate_points_pytorch3d(points=target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :],
                                                                              axes=x_axis,
                                                                              angles=np.pi,
                                                                              translations=torch.zeros(3, device=device).float())
                    target_joints2d_coco = perspective_project_torch(target_joints_coco,
                                                                     None,
                                                                     target_cam_t,
                                                                     focal_length=focal_length,
                                                                     img_wh=img_wh)
                    # Check if joints within image dimensions before cropping + recentering.
                    target_joints2d_vis_coco = check_joints2d_visibility_torch(target_joints2d_coco, img_wh)  # (batch_size, 17)

                    # Render RGB/IUV image
                    lights_rgb_settings = augment_light(batch_size=1,
                                                        device=device,
                                                        light_augment_params=lighting_augment_params)
                    renderer_output = pytorch3d_renderer(vertices=target_vertices_for_rendering,
                                                         textures=texture,
                                                         cam_t=target_cam_t,
                                                         lights_rgb_settings=lights_rgb_settings)
                    iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
                    iuv_in[:, 1:, :, :] = iuv_in[:, 1:, :, :] * 255
                    iuv_in = iuv_in.round()
                    rgb_in = renderer_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)

                    if proxy_rep_augment_params['extreme_crop']:
                        seg_extreme_crop = random_extreme_crop(orig_segs=iuv_in[:, 0, :, :],
                                                               extreme_crop_probability=proxy_rep_augment_params['extreme_crop_probability'])

                    # Crop to person bounding box after bbox scale and centre augmentation
                    if bbox_augment_params['crop_input']:
                        crop_outputs = batch_crop_pytorch_affine(input_wh=(img_wh, img_wh),
                                                                 output_wh=(img_wh, img_wh),
                                                                 num_to_crop=batch_size,
                                                                 device=device,
                                                                 rgb=rgb_in,
                                                                 iuv=iuv_in,
                                                                 joints2D=target_joints2d_coco,
                                                                 bbox_determiner=seg_extreme_crop if proxy_rep_augment_params['extreme_crop'] else None,
                                                                 orig_scale_factor=bbox_augment_params['mean_scale_factor'],
                                                                 delta_scale_range=bbox_augment_params['delta_scale_range'],
                                                                 delta_centre_range=bbox_augment_params['delta_centre_range'],
                                                                 out_of_frame_pad_val=-1)
                        iuv_in = crop_outputs['iuv']
                        target_joints2d_coco = crop_outputs['joints2D']
                        rgb_in = crop_outputs['rgb']

                    # Check if joints within image dimensions after cropping + recentering.
                    target_joints2d_vis_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                               img_wh,
                                                                               vis=target_joints2d_vis_coco)  # (bs, 17)
                    # Check if joints are occluded by the body.
                    if occlude_joints:
                        seg_14_part_occlusion_check = convert_densepose_seg_to_14part_labels(iuv_in[:, 0, :, :])
                        target_joints2d_vis_coco = check_joints2d_occluded_torch(seg_14_part_occlusion_check,
                                                                                 target_joints2d_vis_coco,
                                                                                 pixel_count_threshold=50)  # (bs, 17)

                    # Apply segmentation/IUV-based render augmentations + 2D joints augmentations
                    seg_aug, target_joints2d_coco_input, target_joints2d_vis_coco = augment_proxy_representation(
                        orig_segs=iuv_in[:, 0, :, :],  # Note: out of frame pixels marked with -1
                        orig_joints2D=target_joints2d_coco,
                        proxy_rep_augment_params=proxy_rep_augment_params,
                        orig_joints2D_vis=target_joints2d_vis_coco) # TODO refactor arg names, see augment_rgb

                    # Add background rgb
                    rgb_in = batch_add_rgb_background(backgrounds=background,
                                                      rgb=rgb_in,
                                                      seg=seg_aug)
                    # Apply RGB-based render augmentations + 2D joints augmentations
                    rgb_in, target_joints2d_coco_input, target_joints2d_vis_coco = augment_rgb(rgb=rgb_in,
                                                                                               joints2D=target_joints2d_coco_input,
                                                                                               joints2D_vis=target_joints2d_vis_coco,
                                                                                               rgb_augment_params=rgb_augment_params)
                    # Compute edge-images edges
                    edge_detector_output = edge_detector(rgb_in)
                    edge_in = edge_detector_output['thresholded_thin_edges'] if edge_nms else edge_detector_output['thresholded_grad_magnitude']

                    # Compute 2D joint heatmaps
                    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                               img_wh,
                                                                               std=hmap_gaussian_std)
                    if occlude_joints:
                        j2d_heatmaps = j2d_heatmaps * target_joints2d_vis_coco[:, :, None, None]

                    # Concatenate edge-image and 2D joint heatmaps to create input proxy representation
                    proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)


                with torch.set_grad_enabled(split == 'train'):
                    #############################################################
                    # ---------------------- FORWARD PASS -----------------------
                    #############################################################
                    pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
                    pred_shape_dist, pred_glob, pred_cam_wp = model(proxy_rep_input)
                    # Pose F, U, V and rotmats_mode are (bs, 23, 3, 3) and Pose S is (bs, 23, 3)

                    pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (bs, 3, 3)

                    pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                                       global_orient=pred_glob_rotmats.unsqueeze(1),
                                                       betas=pred_shape_dist.loc,
                                                       pose2rot=False)
                    pred_vertices_mode = pred_smpl_output_mode.vertices  # (bs, 6890, 3)
                    pred_joints_all_mode = pred_smpl_output_mode.joints
                    pred_joints_h36m_mode = pred_joints_all_mode[:, config.ALL_JOINTS_TO_H36M_MAP, :]
                    pred_joints_h36mlsp_mode = pred_joints_h36m_mode[:, config.H36M_TO_J14, :]  # (bs, 14, 3)
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip pred_joints_coco 180° about x-axis so they are right way up when projected
                    pred_joints_coco_mode = aa_rotate_translate_points_pytorch3d(
                        points=pred_joints_all_mode[:, config.ALL_JOINTS_TO_COCO_MAP, :],
                        axes=x_axis,
                        angles=np.pi,
                        translations=torch.zeros(3, device=device).float())
                    pred_joints2d_coco_mode = orthographic_project_torch(pred_joints_coco_mode,
                                                                         pred_cam_wp)  # (bs, 17, 2)

                    with torch.no_grad():
                        pred_reposed_smpl_output_mean = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                                   global_orient=torch.zeros_like(target_pose)[:, :3],
                                                                   betas=pred_shape_dist.loc)
                        pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices  # (bs, 6890, 3)

                    if joints2D_loss_on == 'samples' or joints2D_loss_on == 'means+samples':
                        pred_pose_rotmats_samples = pose_matrix_fisher_sampling_torch(pose_U=pred_pose_U,
                                                                                      pose_S=pred_pose_S,
                                                                                      pose_V=pred_pose_V,
                                                                                      num_samples=num_samples_j2Dloss,
                                                                                      b=1.5,
                                                                                      oversampling_ratio=8,
                                                                                      sample_on_cpu=sample_on_cpu)  # (bs, num samples, 23, 3, 3)
                        pred_shape_samples = pred_shape_dist.rsample([num_samples_j2Dloss]).transpose(0, 1)  # (bs, num_samples, num_smpl_betas)

                        pred_glob_rotmats_expanded = pred_glob_rotmats[:, None, :, :].expand(-1, num_samples_j2Dloss, -1, -1).reshape(-1, 1, 3, 3)  # (bs * num samples, 1, 3, 3)
                        pred_cam_wp_expanded = pred_cam_wp[:, None, :].expand(-1, num_samples_j2Dloss, -1).reshape(-1, 3)  # (bs * num samples, 3)

                        pred_joints_coco_samples = smpl_model(body_pose=pred_pose_rotmats_samples.reshape(-1, 23, 3, 3),  # (bs * num samples, 23, 3, 3)
                                                              global_orient=pred_glob_rotmats_expanded,
                                                              betas=pred_shape_samples.reshape(-1, pred_shape_samples.shape[-1]),  # (bs * num samples, num_smpl_betas)
                                                              pose2rot=False).joints[:, config.ALL_JOINTS_TO_COCO_MAP, :]  # (bs * num samples, 17, 3)
                        # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                        # Need to flip pred_joints_coco_samples 180° about x-axis so they are right way up when projected
                        pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                                        axes=x_axis,
                                                                                        angles=np.pi,
                                                                                        translations=torch.zeros(3, device=device).float())
                        pred_joints2d_coco_samples = orthographic_project_torch(pred_joints_coco_samples,
                                                                                pred_cam_wp_expanded).reshape(-1, num_samples_j2Dloss, pred_joints_coco_samples.shape[1], 2)  # (bs, num samples, 17, 2)
                        if joints2D_loss_on == 'means+samples':
                            pred_joints2d_coco_samples = torch.cat([pred_joints2d_coco_mode[:, None, :, :],
                                                                    pred_joints2d_coco_samples], dim=1)  # (bs, num samples+1, 17, 2)
                    else:
                        pred_joints2d_coco_samples = pred_joints2d_coco_mode[:, None, :, :]  # (batch_size, 1, 17, 2)

                    #############################################################
                    # ----------------- LOSS AND BACKWARD PASS ------------------
                    #############################################################
                    pred_dict_for_loss = {'pose_params_F': pred_pose_F,
                                          'pose_params_U': pred_pose_U,
                                          'pose_params_S': pred_pose_S,
                                          'pose_params_V': pred_pose_V,
                                          'shape_params': pred_shape_dist,
                                          'verts': pred_vertices_mode,
                                          'joints3D': pred_joints_h36mlsp_mode,
                                          'joints2D': pred_joints2d_coco_samples,
                                          'glob_rotmats': pred_glob_rotmats}

                    target_dict_for_loss = {'pose_params_rotmats': target_pose_rotmats,
                                            'shape_params': target_shape,
                                            'verts': target_vertices,
                                            'joints3D': target_joints_h36mlsp,
                                            'joints2D': target_joints2d_coco,
                                            'joints2D_vis': target_joints2d_vis_coco,
                                            'glob_rotmats': target_glob_rotmats}

                    optimiser.zero_grad()
                    loss = criterion(target_dict_for_loss, pred_dict_for_loss)
                    if split == 'train':
                        loss.backward()
                        optimiser.step()

                #############################################################
                # --------------------- TRACK METRICS ----------------------
                #############################################################
                pred_dict_for_loss['joints2D'] = pred_joints2d_coco_mode
                if joints2D_loss_on == 'samples':
                    pred_dict_for_loss['joints2Dsamples'] = pred_joints2d_coco_samples
                elif joints2D_loss_on == 'means+samples':
                    pred_dict_for_loss['joints2Dsamples'] = pred_joints2d_coco_samples[:, 1:, :, :]
                del pred_dict_for_loss['pose_params_F']
                del pred_dict_for_loss['pose_params_U']
                del pred_dict_for_loss['pose_params_S']
                del pred_dict_for_loss['pose_params_V']
                del pred_dict_for_loss['shape_params']
                metrics_tracker.update_per_batch(split=split,
                                                 loss=loss,
                                                 task_losses_dict=None,
                                                 pred_dict=pred_dict_for_loss,
                                                 target_dict=target_dict_for_loss,
                                                 num_inputs_in_batch=batch_size,
                                                 pred_reposed_vertices=pred_reposed_vertices_mean,
                                                 target_reposed_vertices=target_reposed_vertices)

        #############################################################
        # ------------- UPDATE METRICS HISTORY and SAVE -------------
        #############################################################
        metrics_tracker.update_per_epoch()

        save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                best_epoch_val_metrics)

        if save_model_weights_this_epoch:
            for metric in save_val_metrics:
                best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
            print("Best epoch val metrics updated to ", best_epoch_val_metrics)
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print("Best model weights updated!")

        if epoch % epochs_per_save == 0:
            save_dict = {'epoch': epoch,
                         'best_epoch': best_epoch,
                         'best_epoch_val_metrics': best_epoch_val_metrics,
                         'model_state_dict': model.state_dict(),
                         'best_model_state_dict': best_model_wts,
                         'optimiser_state_dict': optimiser.state_dict()}
            torch.save(save_dict,
                       model_save_path + '_epoch{}'.format(epoch) + '.tar')
            print('Model saved! Best Val Metrics:\n',
                  best_epoch_val_metrics,
                  '\nin epoch {}'.format(best_epoch))

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)

    model.load_state_dict(best_model_wts)
    return model
