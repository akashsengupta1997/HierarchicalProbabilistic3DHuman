import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from smplx.lbs import batch_rodrigues

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, batch_crop_opencv_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling, joints2D_error_sorted_verts_sampling


def predict_poseMF_shapeGaussian_net(pose_shape_model,
                                     pose_shape_cfg,
                                     smpl_model,
                                     hrnet_model,
                                     hrnet_cfg,
                                     edge_detect_model,
                                     device,
                                     image_dir,
                                     save_dir,
                                     object_detect_model=None,
                                     joints2Dvisib_threshold=0.75,
                                     visualise_wh=512,
                                     visualise_uncropped=True,
                                     visualise_samples=False):
    """
    Predictor for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on unseen test data.
    Input --> ResNet --> image features --> FC layers --> MF over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    # Setting up body visualisation renderer
    body_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=1,
                                            img_wh=visualise_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            bin_size=32)
    plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)

    hrnet_model.eval()
    pose_shape_model.eval()
    if object_detect_model is not None:
        object_detect_model.eval()
    for image_fname in tqdm(sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])):
        with torch.no_grad():
            # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
            image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_fname)), cv2.COLOR_BGR2RGB)
            orig_image = image.copy()
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 255.0
            # Predict Person Bounding Box + 2D Joints
            hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                         hrnet_config=hrnet_cfg,
                                         object_detect_model=object_detect_model,
                                         image=image,
                                         object_detect_threshold=pose_shape_cfg.DATA.BBOX_THRESHOLD,
                                         bbox_scale_factor=pose_shape_cfg.DATA.BBOX_SCALE_FACTOR)

            # Transform predicted 2D joints and image from HRNet input size to input proxy representation size
            hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                                hrnet_output['cropped_image'].shape[2]]],
                                              dtype=torch.float32,
                                              device=device) * 0.5
            hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                              dtype=torch.float32,
                                              device=device)
            cropped_for_proxy = batch_crop_pytorch_affine(input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
                                                          output_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                          num_to_crop=1,
                                                          device=device,
                                                          joints2D=hrnet_output['joints2D'][None, :, :],
                                                          rgb=hrnet_output['cropped_image'][None, :, :, :],
                                                          bbox_centres=hrnet_input_centre,
                                                          bbox_heights=hrnet_input_height,
                                                          bbox_widths=hrnet_input_height,
                                                          orig_scale_factor=1.0)

            # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
            edge_detector_output = edge_detect_model(cropped_for_proxy['rgb'])
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                             std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
            hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
            hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
            proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
            proxy_rep_input = torch.cat([proxy_rep_img, proxy_rep_heatmaps], dim=1).float()  # (1, 18, img_wh, img_wh)

            # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
            pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

            if pred_glob.shape[-1] == 3:
                pred_glob_rotmats = batch_rodrigues(pred_glob)  # (1, 3, 3)
            elif pred_glob.shape[-1] == 6:
                pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)

            pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                               global_orient=pred_glob_rotmats.unsqueeze(1),
                                               betas=pred_shape_dist.loc,
                                               pose2rot=False)
            pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
            # Need to flip pred_vertices before projecting so that they project the right way up.
            pred_vertices_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                      axes=torch.tensor([1., 0., 0.], device=device),
                                                                      angles=np.pi,
                                                                      translations=torch.zeros(3, device=device))
            # Rotating 90° about vertical axis for visualisation
            pred_vertices_rot90_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                            axes=torch.tensor([0., 1., 0.], device=device),
                                                                            angles=-np.pi / 2.,
                                                                            translations=torch.zeros(3, device=device))
            pred_vertices_rot180_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot90_mode,
                                                                             axes=torch.tensor([0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))
            pred_vertices_rot270_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot180_mode,
                                                                             axes=torch.tensor([0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))

            pred_reposed_smpl_output_mean = smpl_model(betas=pred_shape_dist.loc)
            pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices  # (1, 6890, 3)
            # Need to flip pred_vertices before projecting so that they project the right way up.
            pred_reposed_vertices_flipped_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_mean,
                                                                                      axes=torch.tensor([1., 0., 0.], device=device),
                                                                                      angles=np.pi,
                                                                                      translations=torch.zeros(3, device=device))
            # Rotating 90° about vertical axis for visualisation
            pred_reposed_vertices_rot90_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_flipped_mean,
                                                                                    axes=torch.tensor([0., 1., 0.], device=device),
                                                                                    angles=-np.pi / 2.,
                                                                                    translations=torch.zeros(3, device=device))

            # -------------------------------------- VISUALISATION --------------------------------------
            # Predicted camera corresponding to proxy rep input
            orthographic_scale = pred_cam_wp[:, [0, 0]]
            cam_t = torch.cat([pred_cam_wp[:, 1:],
                               torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            # Estimate per-vertex uncertainty (variance) by sampling SMPL poses/shapes and computing corresponding vertex meshes
            per_vertex_3Dvar, pred_vertices_samples, pred_joints_samples = compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling(
                pose_U=pred_pose_U,
                pose_S=pred_pose_S,
                pose_V=pred_pose_V,
                shape_distribution=pred_shape_dist,
                glob_rotmats=pred_glob_rotmats,
                num_samples=50,
                smpl_model=smpl_model,
                use_mean_shape=True)

            if visualise_samples:
                num_samples = 8
                # Prepare vertex samples for visualisation
                pred_vertices_samples = joints2D_error_sorted_verts_sampling(pred_vertices_samples=pred_vertices_samples,
                                                                             pred_joints_samples=pred_joints_samples,
                                                                             input_joints2D_heatmaps=proxy_rep_input[:, 1:, :, :],
                                                                             pred_cam_wp=pred_cam_wp)[:num_samples, :, :]  # (8, 6890, 3)
                # Need to flip pred_vertices before projecting so that they project the right way up.
                pred_vertices_samples = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples,
                                                                             axes=torch.tensor([1., 0., 0.], device=device),
                                                                             angles=np.pi,
                                                                             translations=torch.zeros(3, device=device))
                pred_vertices_rot90_samples = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples,
                                                                                   axes=torch.tensor([0., 1., 0.], device=device),
                                                                                   angles=-np.pi / 2.,
                                                                                   translations=torch.zeros(3, device=device))

                pred_vertices_samples = torch.cat([pred_vertices_mode, pred_vertices_samples], dim=0)  # (9, 6890, 3)
                pred_vertices_rot90_samples = torch.cat([pred_vertices_rot90_mode, pred_vertices_rot90_samples], dim=0)  # (9, 6890, 3)

            # Generate per-vertex uncertainty colourmap
            vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.cpu().detach().numpy()))[:, :3]
            vertex_var_colours = torch.from_numpy(vertex_var_colours[None, :, :]).to(device).float()

            # Render visualisation outputs
            body_vis_output = body_vis_renderer(vertices=pred_vertices_mode,
                                                cam_t=cam_t,
                                                orthographic_scale=orthographic_scale,
                                                lights_rgb_settings=lights_rgb_settings,
                                                verts_features=vertex_var_colours)
            cropped_for_proxy_rgb = torch.nn.functional.interpolate(cropped_for_proxy['rgb'],
                                                                    size=(visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)
            body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                    rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                    seg=body_vis_output['iuv_images'][:, :, :, 0].round())
            body_vis_rgb = body_vis_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

            body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_vertices_rot90_mode,
                                                   cam_t=fixed_cam_t,
                                                   orthographic_scale=fixed_orthographic_scale,
                                                   lights_rgb_settings=lights_rgb_settings,
                                                   verts_features=vertex_var_colours)['rgb_images'].cpu().detach().numpy()[0]
            body_vis_rgb_rot180 = body_vis_renderer(vertices=pred_vertices_rot180_mode,
                                                    cam_t=fixed_cam_t,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings,
                                                    verts_features=vertex_var_colours)['rgb_images'].cpu().detach().numpy()[0]
            body_vis_rgb_rot270 = body_vis_renderer(vertices=pred_vertices_rot270_mode,
                                                    textures=plain_texture,
                                                    cam_t=fixed_cam_t,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings,
                                                    verts_features=vertex_var_colours)['rgb_images'].cpu().detach().numpy()[0]

            # Reposed body visualisation
            reposed_body_vis_rgb = body_vis_renderer(vertices=pred_reposed_vertices_flipped_mean,
                                                     textures=plain_texture,
                                                     cam_t=fixed_cam_t,
                                                     orthographic_scale=fixed_orthographic_scale,
                                                     lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
            reposed_body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_reposed_vertices_rot90_mean,
                                                           textures=plain_texture,
                                                           cam_t=fixed_cam_t,
                                                           orthographic_scale=fixed_orthographic_scale,
                                                           lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

            # Combine all visualisations
            combined_vis_rows = 2
            combined_vis_cols = 4
            combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                        dtype=body_vis_rgb.dtype)
            # Cropped input image
            combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

            # Proxy representation + 2D joints scatter + 2D joints confidences
            proxy_rep_input = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
            proxy_rep_input = np.stack([proxy_rep_input]*3, axis=-1)  # single-channel to RGB
            proxy_rep_input = cv2.resize(proxy_rep_input, (visualise_wh, visualise_wh))
            for joint_num in range(cropped_for_proxy['joints2D'].shape[1]):
                hor_coord = cropped_for_proxy['joints2D'][0, joint_num, 0].item() * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                ver_coord = cropped_for_proxy['joints2D'][0, joint_num, 1].item() * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                cv2.circle(proxy_rep_input,
                           (int(hor_coord), int(ver_coord)),
                           radius=3,
                           color=(255, 0, 0),
                           thickness=-1)
                cv2.putText(proxy_rep_input,
                            str(joint_num),
                            (int(hor_coord + 4), int(ver_coord + 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                cv2.putText(proxy_rep_input,
                            str(joint_num) + " {:.2f}".format(hrnet_output['joints2Dconfs'][joint_num].item()),
                            (10, 16 * (joint_num + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
            combined_vis_fig[visualise_wh:2*visualise_wh, :visualise_wh] = proxy_rep_input

            # Posed 3D body
            combined_vis_fig[:visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb
            combined_vis_fig[visualise_wh:2*visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb_rot90
            combined_vis_fig[:visualise_wh, 2*visualise_wh:3*visualise_wh] = body_vis_rgb_rot180
            combined_vis_fig[visualise_wh:2*visualise_wh, 2*visualise_wh:3*visualise_wh] = body_vis_rgb_rot270

            # T-pose 3D body
            combined_vis_fig[:visualise_wh, 3*visualise_wh:4*visualise_wh] = reposed_body_vis_rgb
            combined_vis_fig[visualise_wh:2*visualise_wh, 3*visualise_wh:4*visualise_wh] = reposed_body_vis_rgb_rot90
            vis_save_path = os.path.join(save_dir, image_fname)
            cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)

            if visualise_uncropped:
                # Uncropped visualisation by projecting 3D body onto original image
                rgb_to_uncrop = body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy()
                iuv_to_uncrop = body_vis_output['iuv_images'].permute(0, 3, 1, 2).contiguous().cpu().detach().numpy()
                bbox_centres = hrnet_output['bbox_centre'][None].cpu().detach().numpy()
                bbox_whs = torch.max(hrnet_output['bbox_height'], hrnet_output['bbox_width'])[None].cpu().detach().numpy()
                bbox_whs *= pose_shape_cfg.DATA.BBOX_SCALE_FACTOR
                uncropped_for_visualise = batch_crop_opencv_affine(output_wh=(visualise_wh, visualise_wh),
                                                                   num_to_crop=1,
                                                                   rgb=rgb_to_uncrop,
                                                                   iuv=iuv_to_uncrop,
                                                                   bbox_centres=bbox_centres,
                                                                   bbox_whs=bbox_whs,
                                                                   uncrop=True,
                                                                   uncrop_wh=(orig_image.shape[1], orig_image.shape[0]))
                uncropped_rgb = uncropped_for_visualise['rgb'][0].transpose(1, 2, 0) * 255
                uncropped_seg = uncropped_for_visualise['iuv'][0, 0, :, :]
                background_pixels = uncropped_seg[:, :, None] == 0  # Body pixels are > 0
                uncropped_rgb_with_background = uncropped_rgb * (np.logical_not(background_pixels)) + \
                                                orig_image * background_pixels

                uncropped_vis_save_path = os.path.splitext(vis_save_path)[0] + '_uncrop.png'
                cv2.imwrite(uncropped_vis_save_path, uncropped_rgb_with_background[:, :, ::-1])

            if visualise_samples:
                samples_rows = 3
                samples_cols = 6
                samples_fig = np.zeros((samples_rows * visualise_wh, samples_cols * visualise_wh, 3),
                                       dtype=body_vis_rgb.dtype)
                for i in range(num_samples + 1):
                    body_vis_output_sample = body_vis_renderer(vertices=pred_vertices_samples[[i]],
                                                               textures=plain_texture,
                                                               cam_t=cam_t,
                                                               orthographic_scale=orthographic_scale,
                                                               lights_rgb_settings=lights_rgb_settings)
                    body_vis_rgb_sample = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                                   rgb=body_vis_output_sample['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                                   seg=body_vis_output_sample['iuv_images'][:, :, :, 0].round())
                    body_vis_rgb_sample = body_vis_rgb_sample.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    body_vis_rgb_rot90_sample = body_vis_renderer(vertices=pred_vertices_rot90_samples[[i]],
                                                                  textures=plain_texture,
                                                                  cam_t=fixed_cam_t,
                                                                  orthographic_scale=fixed_orthographic_scale,
                                                                  lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

                    row = (2 * i) // samples_cols
                    col = (2 * i) % samples_cols
                    samples_fig[row*visualise_wh:(row+1)*visualise_wh, col*visualise_wh:(col+1)*visualise_wh] = body_vis_rgb_sample

                    row = (2 * i + 1) // samples_cols
                    col = (2 * i + 1) % samples_cols
                    samples_fig[row * visualise_wh:(row + 1) * visualise_wh, col * visualise_wh:(col + 1) * visualise_wh] = body_vis_rgb_rot90_sample

                    samples_fig_save_path = os.path.splitext(vis_save_path)[0] + '_samples.png'
                    cv2.imwrite(samples_fig_save_path, samples_fig[:, :, ::-1] * 255)







