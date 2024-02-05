import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from smplx.lbs import batch_rodrigues
from PIL import Image

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, batch_crop_opencv_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling, joints2D_error_sorted_verts_sampling

from utils.rigid_transform_utils import matrix_to_axis_angle

from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch, convert_densepose_seg_to_14part_labels, \
    ALL_JOINTS_TO_H36M_MAP, ALL_JOINTS_TO_COCO_MAP, H36M_TO_J14

from utils.cam_utils import perspective_project_torch, orthographic_project_torch


def optim_real_data(pose_shape_model,
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
                                visualise_samples=False,
                                silh_opt_renderer=None,
                                verbose=False):
    #print(verbose)
    x_axis = torch.tensor([1., 0., 0.],
                    device=device, dtype=torch.float32)
        
    batch_size=4


    # Setting up body visualisation renderer
    body_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=batch_size,
                                            img_wh=visualise_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            bin_size=32)
    opt_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=batch_size,
                                            img_wh=visualise_wh,
                                            projection_type='perspective',
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



    target_silhs = torch.empty((0, 1, 256, 256)).to(device)
    images = torch.empty((0, 3, 256, 256)).to(device)
    proxy_inputs = torch.empty((0, 18, pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE)).to(device) #(N+1, 18, 256, 256)
    target_joints2d = torch.empty((0, 17, 2)).to(device)

    image_history = []


    for image_fname in (sorted([f for f in os.listdir(image_dir) if f.endswith(('Apose.jpg'))])):
        with torch.no_grad():
            print(image_fname)
            # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
            image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_fname)), cv2.COLOR_BGR2RGB)
            
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 255.0

            #target_seg = np.load(os.path.join('./my_data/segs_npy', (os.path.splitext(image_fname)[0] + '_seg.npy')))
            # target_seg[target_seg!=0] = 1
            # target_silh = torch.from_numpy(np.rot90(target_seg, 3).copy()).unsqueeze(0).float().to(device)
            print(image.shape)

            target_seg = np.load(os.path.join('./my_data/sam_npy', (os.path.splitext(image_fname)[0] + '.npy')))
            print(target_seg.shape)
            target_silh = torch.from_numpy(target_seg).unsqueeze(0).float().to(device)


            hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                            hrnet_config=hrnet_cfg,
                                            object_detect_model=object_detect_model,
                                            image=image,
                                            silh=target_silh,
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

            # for batch_crop always input bchw
            cropped_for_proxy = batch_crop_pytorch_affine(input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
                                                            output_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                            num_to_crop=1,
                                                            device=device,
                                                            joints2D=hrnet_output['joints2D'][None, :, :],
                                                            rgb=hrnet_output['cropped_image'][None, :, :, :],
                                                            silh=hrnet_output['cropped_silh'][None, :, :, :],
                                                            bbox_centres=hrnet_input_centre,
                                                            bbox_heights=hrnet_input_height,
                                                            bbox_widths=hrnet_input_height,
                                                            orig_scale_factor=1.0)
            
            target_joints2d = torch.cat([target_joints2d, cropped_for_proxy['joints2D']], dim=0)

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
            
            target_silhs= torch.cat((target_silhs, cropped_for_proxy['silh']))
            images = torch.cat((images, cropped_for_proxy['rgb']))
            proxy_inputs = torch.cat([proxy_inputs, proxy_rep_input], dim=0)

            print(proxy_inputs.shape)

    with torch.no_grad():
    # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
        pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
        pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_inputs)
        # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

        if pred_glob.shape[-1] == 3:
            pred_glob_rotmats = batch_rodrigues(pred_glob)  # (N, 3, 3)
        elif pred_glob.shape[-1] == 6:
            pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (N, 3, 3)
        
        
        pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                                       global_orient=pred_glob_rotmats.unsqueeze(1),
                                                       betas=pred_shape_dist.loc,
                                                       pose2rot=False)
        pred_vertices_mode = pred_smpl_output_mode.vertices  # (bs, 6890, 3)
        pred_joints_all_mode = pred_smpl_output_mode.joints
        pred_joints_coco_mode = aa_rotate_translate_points_pytorch3d(
                        points=pred_joints_all_mode[:, ALL_JOINTS_TO_COCO_MAP, :],
                        axes=x_axis,
                        angles=np.pi,
                        translations=torch.zeros(3, device=device).float())
        pred_joints2d_coco_mode = orthographic_project_torch(pred_joints_coco_mode,
                                                                pred_cam_wp)  # (bs, 17, 2)
                    
    target_joints2D = (2.0 * target_joints2d) / 256.0 - 1.0 


    lr=0.002
    flag=True
    
    num_iters = 1000
    #pose_iters_1 = 1400
    num_iters = 2000
    pose_iters_1 = 3000


    loss_history = {'silhouette': [], 'joint':[], 'one':[], 'two':[], 'three':[], 'four':[],}

    shape = pred_shape_dist.loc.clone().mean(dim=(0))
    pose = matrix_to_axis_angle(pred_pose_rotmats_mode.clone())
    glob_rot = matrix_to_axis_angle(pred_glob_rotmats.clone())

    #cam_focal_len = torch.ones(pred_cam_wp.shape[0], 1, device=device) * 300.
    cam_focal_len = torch.ones(1, device=device) * 300.
    cam_t = torch.cat([pred_cam_wp[:, 1:],
                    torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                    dim=-1).to(device)   # (N, tx, ty, 2.5)
    

    cam_t.requires_grad=True
    cam_focal_len.requires_grad=True
    shape.requires_grad=True
    pose.requires_grad=True
    glob_rot.requires_grad=True

    print('---')
    print(shape)
    print('---')

    opt_variables = [pose, glob_rot, shape, cam_t, cam_focal_len]
    optimiser = torch.optim.SGD(opt_variables, lr=lr)

    for iter in tqdm(range(num_iters)):
        if iter == pose_iters_1:
            #for g in optimiser.param_groups:
            #    g['lr'] = 0.02
            pose.requires_grad = False
            cam_focal_len.requires_grad = False
            cam_t.requires_grad = True
            glob_rot.requires_grad=False
            print(f'shape only: {shape}')

        opt_smpl = smpl_model(body_pose=pose,
                                        global_orient=glob_rot.unsqueeze(1),
                                        betas=shape.expand(batch_size, -1),
                                        pose2rot=True) # if true, assume input are axis-angle form
        opt_verts = opt_smpl.vertices  # (bs, 6890, 3)
        opt_verts = aa_rotate_translate_points_pytorch3d(points=opt_verts,
                                                                axes=x_axis,
                                                                angles=np.pi,
                                                                translations=torch.zeros(3, device=device).float())

        opt_silhs = silh_opt_renderer(vertices=opt_verts,
                            cam_t=cam_t,
                            perspective_focal_length=cam_focal_len)['silhouettes']
        
        if verbose and flag:
            pred_silh_tosave = opt_silhs.clone().detach().cpu().numpy()
            pred_silh_name = os.path.join(save_dir, (os.path.splitext(image_fname)[0] + "_init_pred_silh.png"))
            cv2.imwrite(pred_silh_name, pred_silh_tosave[0, :, :] * 255)
            flag=False
            print(target_silhs.shape, opt_silhs.shape)


        # Silhouette Loss
        silh_loss = ((target_silhs[:, 0, :, :] - opt_silhs) ** 2).mean(dim=(1,2))
        joint_loss = ((target_joints2D - pred_joints2d_coco_mode) ** 2).mean()
        final_loss = silh_loss.sum() + joint_loss
        #silh_loss = ((target_silhs - opt_silhs) ** 2).mean()
        #final_loss = silh_loss


        optimiser.zero_grad()
        loss = final_loss
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            loss_history['silhouette'].append(final_loss.item())
            loss_history['joint'].append(silh_loss[0].item())
            loss_history['one'].append(silh_loss[0].item())
            loss_history['two'].append(silh_loss[1].item())
            loss_history['three'].append(silh_loss[2].item())
            loss_history['four'].append(silh_loss[3].item())
        
        per_vertex_3Dvar = torch.zeros((6890,))+0.1
        vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
        vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.numpy()))[:, :3]
        vertex_var_colours = torch.from_numpy(vertex_var_colours[None, :, :]).expand(batch_size, -1, -1).to(device).float()

        if iter % 5 == 0:
            progress_rendered = opt_vis_renderer(vertices=opt_verts,
                                            cam_t=cam_t,
                                            lights_rgb_settings=lights_rgb_settings,
                                            perspective_focal_length=600,
                                            verts_features=vertex_var_colours)
            
            cropped_for_proxy_rgb = torch.nn.functional.interpolate(target_silhs.expand(-1, 3, -1, -1),
                                                                    size=(visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)

            progress_vis = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                    rgb=progress_rendered['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                    seg=progress_rendered['iuv_images'][:, :, :, 0].round())
            progress_vis = progress_vis.cpu().detach().numpy().transpose(0, 2, 3, 1)

            image_history.append((progress_vis*255).astype(np.uint8))

    image_history = np.array(image_history)

    #print(np.array(image_history).shape)

    # Optimisation gif visualisation
    import imageio
    for i in range(4):
        gif_save_path = os.path.join(save_dir, f'{str(i)}_test.gif')
        imageio.mimsave(gif_save_path, image_history[:, i, :, :, :], duration=0.15)


    print(" ")
    print('---')
    #print(cam_t)
    print(shape)
    print('---')


    plt.figure()
    plt.subplot(121)
    plt.title('Silhouette Loss')
    plt.xlabel("Iterations")
    plt.plot(loss_history['silhouette'])
    plt.plot(loss_history['joint'])
    plt.plot(loss_history['one'])
    plt.plot(loss_history['two'])
    plt.plot(loss_history['three'])
    plt.plot(loss_history['four'])
    plt.legend(['all', 'joint', '1', '2', '3', '4'])
    plt.savefig(os.path.join(save_dir, 'sil_loss.png'))
    plt.close()
    plt.figure()
    plt.subplot(121)
    plt.title('Silhouette Loss')
    plt.xlabel("Iterations")
    plt.plot(loss_history['silhouette'][pose_iters_1:])
    plt.plot(loss_history['one'][pose_iters_1:])
    plt.plot(loss_history['two'][pose_iters_1:])
    plt.plot(loss_history['three'][pose_iters_1:])
    plt.plot(loss_history['four'][pose_iters_1:])
    plt.legend(['all', '1', '2', '3', '4'])
    plt.savefig(os.path.join(save_dir, 'shape_sil_loss.png'))
    plt.close()

    shape = shape.expand(batch_size, -1)

    shape_fname = os.path.join(save_dir, os.path.splitext(image_fname)[0] + '_betas.npy')
    np.save(shape_fname, shape.detach().cpu().numpy())

    if verbose:
        silh_tosave = target_silh.detach().cpu().numpy()
        pred_silh_tosave = opt_silhs.detach().cpu().numpy()

        silh_name = os.path.join(save_dir, os.path.splitext(image_fname)[0] + "_target_silh.png") 
        pred_silh_name = os.path.join(save_dir, os.path.splitext(image_fname)[0] + "_final_pred_silh.png") 
        cv2.imwrite(silh_name, silh_tosave[0, :, :] * 255)
        cv2.imwrite(pred_silh_name, pred_silh_tosave[0, :, :] * 255)

    plt.figure()
    img = plt.imshow(opt_silhs[0].detach().cpu().numpy())
    plt.colorbar(img)
    plt.savefig(os.path.join(save_dir, 'vis_silh.png'))

    with torch.no_grad():
        pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                            global_orient=pred_glob_rotmats.unsqueeze(1),
                                            betas=pred_shape_dist.loc,
                                            pose2rot=False)
        baseline_verts = pred_smpl_output_mode.vertices  # (1, 6890, 3)
        # Need to flip opt_verts before projecting so that they project the right way up.
        baseline_verts = aa_rotate_translate_points_pytorch3d(points=baseline_verts,
                                                                    axes=torch.tensor([1., 0., 0.], device=device),
                                                                    angles=np.pi,
                                                                    translations=torch.zeros(3, device=device))

        opt_smpl = smpl_model(body_pose=pose,
                                        global_orient=glob_rot.unsqueeze(1),
                                        betas=shape,
                                        pose2rot=True) # if true, assume input are axis-angle form
        opt_verts = opt_smpl.vertices  # (bs, 6890, 3)
        opt_verts = aa_rotate_translate_points_pytorch3d(points=opt_verts,
                                                                    axes=x_axis,
                                                                    angles=np.pi,
                                                                    translations=torch.zeros(3, device=device).float())

        opt_silhs = silh_opt_renderer(vertices=opt_verts,
                            cam_t=cam_t,
                            perspective_focal_length=cam_focal_len)['silhouettes']
        

        per_vertex_3Dvar = torch.zeros((6890,))+0.1
        vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
        vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.numpy()))[:, :3]
        vertex_var_colours = torch.from_numpy(vertex_var_colours[None, :, :]).expand(batch_size, -1, -1).to(device).float()


        opt_vis = opt_vis_renderer(vertices=opt_verts,
                                        cam_t=cam_t,
                                        lights_rgb_settings=lights_rgb_settings,
                                        perspective_focal_length=600,
                                        verts_features=vertex_var_colours)
        
        cropped_for_proxy_rgb = torch.nn.functional.interpolate(target_silhs.expand(-1, 3, -1, -1)*255,
                                                                size=(visualise_wh, visualise_wh),
                                                                mode='bilinear',
                                                                align_corners=False)

        opt_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                rgb=opt_vis['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                seg=opt_vis['iuv_images'][:, :, :, 0].round())
        opt_vis_rgb = opt_vis_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)


        # Predicted camera corresponding to proxy rep input
        orthographic_scale = pred_cam_wp[:, [0, 0]]
        cam_t = torch.cat([pred_cam_wp[:, 1:],
                            torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                            dim=-1)

        # Render visualisation outputs
        body_vis_output = body_vis_renderer(vertices=baseline_verts,
                                            cam_t=cam_t,
                                            orthographic_scale=orthographic_scale,
                                            lights_rgb_settings=lights_rgb_settings,
                                            verts_features=vertex_var_colours)
        body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                seg=body_vis_output['iuv_images'][:, :, :, 0].round())
        body_vis_rgb = body_vis_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)


        # Combine all visualisations
        combined_vis_rows = 2
        combined_vis_cols = 4
        combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                    dtype=body_vis_rgb.dtype)
        # Cropped input image
        combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)


        for i in range(0, batch_size):
            baseline_tovis = cv2.resize(body_vis_rgb[i], (visualise_wh, visualise_wh))
            combined_vis_fig[:visualise_wh, i * visualise_wh: (i+1)*visualise_wh] = baseline_tovis

            opt_tovis = cv2.resize(opt_vis_rgb[i], (visualise_wh, visualise_wh))
            #opt_tovis = cv2.resize((240*opt_silhs.unsqueeze(1).expand(-1, 3, -1, -1)[i]).detach().cpu().numpy(),(visualise_wh, visualise_wh))
            combined_vis_fig[visualise_wh:2*visualise_wh, i*visualise_wh:(i+1)*visualise_wh] = opt_tovis
        vis_save_path = os.path.join(save_dir, 'mult_result.png')
        cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)

        
