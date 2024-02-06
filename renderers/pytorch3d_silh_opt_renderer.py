import torch
import torch.nn as nn
import numpy as np

#import config

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    OrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    BlendParams)


class SilhouetteRenderer(nn.Module):
    def __init__(self,
                 device,
                 batch_size,
                 smpl_faces,
                 img_wh=256,
                 cam_t=None,
                 cam_R=None,
                 projection_type='perspective',
                 perspective_focal_length=300,
                 orthographic_scale=0.9,
                 blur_radius=None,
                 faces_per_pixel=20,
                 bin_size=None,
                 max_faces_per_bin=None,
                 perspective_correct=False,
                 cull_backfaces=False,
                 clip_barycentric_coords=None,
                 blend_sigma=1e-4,
                 blend_gamma=1e-4,
                 background_color=(0.0, 0.0, 0.0)):
        
        super().__init__()
        self.img_wh = img_wh
        self.smpl_faces = smpl_faces

        # ---------- Cameras ----------
        # Pre-defined here but can be specified in forward pass if cameras will be optimised
        assert projection_type in ['perspective', 'orthographic'], print('Invalid projection type:', projection_type)
        print('Renderer projection type:', projection_type)
        self.projection_type = projection_type
        if cam_R is None:
            # Rotating 180° about z-axis to make pytorch3d camera convention same as what I've been using so far in my perspective_project_torch/NMR/pyrender
            # (Actually pyrender also has a rotation defined in the renderer to make it same as NMR.)
            cam_R = torch.tensor([[-1., 0., 0.],
                                  [0., -1., 0.],
                                  [0., 0., 1.]], device=device).float()
            cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
        if cam_t is None:
            cam_t = torch.tensor([0., 0.2, 2.5]).float().to(device)[None, :].expand(batch_size, -1)
        # Pytorch3D camera is rotated 180° about z-axis to match my perspective_project_torch/NMR's projection convention.
        # So, need to also rotate the given camera translation (implemented below as elementwise-mul).
        cam_t = cam_t * torch.tensor([-1., -1., 1.], device=cam_t.device).float()
        if projection_type == 'perspective':
            self.cameras = PerspectiveCameras(device=device,
                                              R=cam_R,
                                              T=cam_t,
                                              focal_length=perspective_focal_length,
                                              principal_point=((img_wh/2., img_wh/2.),),
                                              image_size=((img_wh, img_wh),))
        elif projection_type == 'orthographic':
            self.cameras = OrthographicCameras(device=device,
                                               R=cam_R,
                                               T=cam_t,
                                               focal_length=orthographic_scale*(img_wh/2.),
                                               principal_point=((img_wh / 2., img_wh / 2.),),
                                               image_size=((img_wh, img_wh),))

        # ---------- Rasterizer ----------
        if blur_radius is None:
            blur_radius = np.log(1. / 1e-4 - 1.) * blend_sigma
        raster_settings = RasterizationSettings(image_size=img_wh,
                                                blur_radius=blur_radius,###
                                                faces_per_pixel=faces_per_pixel,
                                                bin_size=bin_size,
                                                max_faces_per_bin=max_faces_per_bin,
                                                perspective_correct=perspective_correct,
                                                cull_backfaces=cull_backfaces,
                                                clip_barycentric_coords=clip_barycentric_coords)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.rasterizer.to(device)

        # ---------- Shader ----------
        blend_params = BlendParams(background_color=background_color,
                                   sigma=blend_sigma,
                                   gamma=blend_gamma)
        self.silh_shader = SoftSilhouetteShader(blend_params=blend_params) ####

    def forward(self,
                vertices,
                cam_t=None,
                perspective_focal_length=None,
                orthographic_scale=None):
        
        if cam_t is not None:
            # Pytorch3D camera is rotated 180° about z-axis to match my perspective_project_torch/NMR's projection convention.
            # So, need to also rotate the given camera translation (implemented below as elementwise-mul).
            self.cameras.T = cam_t * torch.tensor([-1., -1., 1.], device=cam_t.device).float()
        if perspective_focal_length is not None and self.projection_type == 'perspective':
            self.cameras.focal_length = perspective_focal_length
        if orthographic_scale is not None and self.projection_type == 'orthographic':
            self.cameras.focal_length = orthographic_scale * (self.img_wh / 2.0)

        # Rasterize
        meshes = Meshes(verts=vertices, faces=self.smpl_faces)
        fragments = self.rasterizer(meshes, cameras=self.cameras)

        # Render silhouette
        output = {}
        silhouette = self.silh_shader(fragments, meshes)[:, :, :, 3]
        if self.rasterizer.raster_settings.blur_radius == 0:
            silhouette[silhouette > 1e-2] = 1.
        output['silhouettes'] = silhouette

        return output