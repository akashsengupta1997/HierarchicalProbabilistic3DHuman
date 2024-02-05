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
        """
        :param img_wh: Size of rendered image.
        :param blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary.
            Set to 0 (no blur) if rendering for visualisation purposes.
        :param faces_per_pixel: Number of faces to save per pixel, returning
            the nearest faces_per_pixel points along the z-axis.
            Set to 1 if rendering for visualisation purposes.
        :param bin_size: Size of bins to use for coarse-to-fine rasterization (i.e.
            breaking image into tiles with size=bin_size before rasterising?).
            Setting bin_size=0 uses naive rasterization; setting bin_size=None
            attempts to set it heuristically based on the shape of the input (i.e. image_size).
            This should not affect the output, but can affect the speed of the forward pass.
            Heuristic based formula maps image_size -> bin_size as follows:
                image_size < 64 -> 8
                16 < image_size < 256 -> 16
                256 < image_size < 512 -> 32
                512 < image_size < 1024 -> 64
                1024 < image_size < 2048 -> 128
        :param max_faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of faces allowed within each
            bin. If more than this many faces actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
            the memory usage in the forward pass.
            Heuristic used if None value given:
                max_faces_per_bin = int(max(10000, meshes._F / 5))
        :param perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels.
        :param cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        :param clip_barycentric_coords: By default, turn on clip_barycentric_coords if blur_radius > 0.
        When blur_radius > 0, a face can be matched to a pixel that is outside the face,
        resulting in negative barycentric coordinates.
        """
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
        """
        Render a batch of silhouette images from a batch of meshes.

        Fragments output from rasterizer:
        pix_to_face:
          LongTensor of shape (B, image_size, image_size, faces_per_pixel)
          specifying the indices of the faces (in the packed faces) which overlap each pixel in the image.
        zbuf:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel)
          giving the z-coordinates of the nearest faces at each pixel in world coordinates, sorted in ascending z-order.
        bary_coords:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the nearest faces at each pixel, sorted in ascending z-order.
        pix_dists:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the x/y plane of each point closest to the pixel.

        :param vertices: (B, N, 3)
        :param cam_t: (B, 3)
        :param perspective_focal_length: (B, 1)
        :param orthographic_scale: (B, 2)
        :param lights_rgb_settings: dict of lighting settings with location, ambient_color, diffuse_color and specular_color.
        :returns rgb_images: (B, img_wh, img_wh, 3)
        :returns iuv_images: (B, img_wh, img_wh, 3) IUV images give bodypart (I) + UV coordinate information. Parts are DP convention, indexed 1-24.
        :returns depth_images: (B, img_wh, img_wh)
        """
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