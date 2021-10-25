import torch
import numpy as np

from utils.rigid_transform_utils import quat_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_heatmaps_to_2Djoints_coordinates_torch, ALL_JOINTS_TO_COCO_MAP


def bingham_sampling_for_matrix_fisher_torch(A,
                                             num_samples,
                                             Omega=None,
                                             Gaussian_std=None,
                                             b=1.5,
                                             M_star=None,
                                             oversampling_ratio=8):
    """
    Sampling from a Bingham distribution with 4x4 matrix parameter A.
    Here we assume that A is a diagonal matrix (needed for matrix-Fisher sampling).
    Bing(A) is simulated by rejection sampling from ACG(I + 2A/b) (since ACG > Bingham everywhere).
    Rejection sampling is batched + differentiable (using re-parameterisation trick).

    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

    :param A: (4,) tensor parameter of Bingham distribution on 3-sphere.
        Represents the diagonal of a 4x4 diagonal matrix.
    :param num_samples: scalar. Number of samples to draw.
    :param Omega: (4,) Optional tensor parameter of ACG distribution on 3-sphere.
    :param Gaussian_std: (4,) Optional tensor parameter (standard deviations) of diagonal Gaussian in R^4.
    :param num_samples:
    :param b: Hyperparameter for rejection sampling using envelope ACG distribution with
        Omega = I + 2A/b
    :param oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.
    :return: samples: (num_samples, 4) and accept_ratio
    """
    assert A.shape == (4,)
    assert A.min() >= 0

    if Omega is None:
        Omega = torch.ones(4, device=A.device) + 2*A/b  # Will sample from ACG(Omega) with Omega = I + 2A/b.
    if Gaussian_std is None:
        Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    if M_star is None:
        M_star = np.exp(-(4 - b) / 2) * ((4 / b) ** 2)  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    samples_obtained = False
    while not samples_obtained:
        eps = torch.randn(num_samples * oversampling_ratio, 4, device=A.device).float()
        y = Gaussian_std * eps
        samples = y / torch.norm(y, dim=1, keepdim=True)  # (num_samples * oversampling_ratio, 4)

        with torch.no_grad():
            p_Bing_star = torch.exp(-torch.einsum('bn,n,bn->b', samples, A, samples))  # (num_samples * oversampling_ratio,)
            p_ACG_star = torch.einsum('bn,n,bn->b', samples, Omega, samples) ** (-2)  # (num_samples * oversampling_ratio,)
            # assert torch.all(p_Bing_star <= M_star * p_ACG_star + 1e-6)

            w = torch.rand(num_samples * oversampling_ratio, device=A.device)
            accept_vector = w < p_Bing_star / (M_star * p_ACG_star)  # (num_samples * oversampling_ratio,)
            num_accepted = accept_vector.sum().item()
        if num_accepted >= num_samples:
            samples = samples[accept_vector, :]  # (num_accepted, 4)
            samples = samples[:num_samples, :]  # (num_samples, 4)
            samples_obtained = True
            accept_ratio = num_accepted / num_samples * 4
        else:
            print('Failed sampling. {} samples accepted, {} samples required.'.format(num_accepted, num_samples))

    return samples, accept_ratio


def pose_matrix_fisher_sampling_torch(pose_U,
                                      pose_S,
                                      pose_V,
                                      num_samples,
                                      b=1.5,
                                      oversampling_ratio=8,
                                      sample_on_cpu=False):
    """
    Sampling from matrix-Fisher distributions defined over SMPL joint rotation matrices.
    MF distribution is simulated by sampling quaternions Bingham distribution (see above) and
    converting quaternions to rotation matrices.

    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

    :param pose_U: (B, 23, 3, 3)
    :param pose_S: (B, 23, 3)
    :param pose_V: (B, 23, 3, 3)
    :param num_samples: scalar. Number of samples to draw.
    :param b: Hyperparameter for rejection sampling using envelope ACG distribution.
    :param oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.
    :param sample_on_cpu: do sampling on CPU instead of GPU.
    :return: R_samples: (B, num samples, 23, 3, 3)
    """
    batch_size = pose_U.shape[0]
    num_joints = pose_U.shape[1]

    # Proper SVD
    with torch.no_grad():
        detU, detV = torch.det(pose_U.detach().cpu()).to(pose_U.device), torch.det(pose_V.detach().cpu()).to(pose_V.device)
    pose_U_proper = pose_U.clone()
    pose_S_proper = pose_S.clone()
    pose_V_proper = pose_V.clone()
    pose_S_proper[:, :, 2] *= detU * detV  # Proper singular values: s3 = s3 * det(UV)
    pose_U_proper[:, :, :, 2] *= detU.unsqueeze(-1)  # Proper U = U diag(1, 1, det(U))
    pose_V_proper[:, :, :, 2] *= detV.unsqueeze(-1)

    # Sample quaternions from Bingham(A)
    if sample_on_cpu:
        sample_device = 'cpu'
    else:
        sample_device = pose_S_proper.device
    bingham_A = torch.zeros(batch_size, num_joints, 4, device=sample_device)
    bingham_A[:, :, 1] = 2 * (pose_S_proper[:, :, 1] + pose_S_proper[:, :, 2])
    bingham_A[:, :, 2] = 2 * (pose_S_proper[:, :, 0] + pose_S_proper[:, :, 2])
    bingham_A[:, :, 3] = 2 * (pose_S_proper[:, :, 0] + pose_S_proper[:, :, 1])

    Omega = torch.ones(batch_size, num_joints, 4, device=bingham_A.device) + 2 * bingham_A / b  # Will sample from ACG(Omega) with Omega = I + 2A/b.
    Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    M_star = np.exp(-(4 - b) / 2) * ((4 / b) ** 2)  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    pose_quat_samples_batch = torch.zeros(batch_size, num_samples, num_joints, 4, device=pose_U.device).float()
    for i in range(batch_size):
        for joint in range(num_joints):
            quat_samples, accept_ratio = bingham_sampling_for_matrix_fisher_torch(A=bingham_A[i, joint, :],
                                                                                  num_samples=num_samples,
                                                                                  Omega=Omega[i, joint, :],
                                                                                  Gaussian_std=Gaussian_std[i, joint, :],
                                                                                  b=b,
                                                                                  M_star=M_star,
                                                                                  oversampling_ratio=oversampling_ratio)
            pose_quat_samples_batch[i, :, joint, :] = quat_samples

    pose_R_samples_batch = quat_to_rotmat(quat=pose_quat_samples_batch.view(-1, 4)).view(batch_size, num_samples, num_joints, 3, 3)
    pose_R_samples_batch = torch.matmul(pose_U_proper[:, None, :, :, :],
                                        torch.matmul(pose_R_samples_batch, pose_V_proper[:, None, :, :, :].transpose(dim0=-1, dim1=-2)))

    return pose_R_samples_batch


def compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling(pose_U,
                                                                  pose_S,
                                                                  pose_V,
                                                                  shape_distribution,
                                                                  glob_rotmats,
                                                                  num_samples,
                                                                  smpl_model,
                                                                  use_mean_shape=False):
    """
    Uncertainty = Per-vertex average Euclidean distance from the mean (computed from samples)
    Sampling procedure:
        1) Sample SMPL betas from shape distribution.
        2) Sample pose rotation matrices from pose distribution - matrix fisher M(USV^T)
        3) Pass sampled betas and rotation matrices to SMPL to get full shaped + posed body sample.
    Batch size should be 1 for pose USV and reposed_vertices_distribution.
    :param pose_U: Tensor, (1, 23, 3, 3)
    :param pose_S: Tensor, (1, 23, 3)
    :param pose_V: Tensor, (1, 23, 3, 3)
    :param shape_distribution: torch Normal distribution
    :param glob_rotmats: Tensor (B, 3, 3)
    :param num_samples: int, number of samples to draw
    :param use_mean_shape: bool, use mean shape for samples?
    :return avg_vertices_distance_from_mean: Array, (6890,), average Euclidean distance from mean for each vertex.
    :return vertices_samples
    """
    assert pose_U.shape[0] == pose_S.shape[0] == pose_V.shape[0] == 1  # batch size should be 1
    pose_sample_rotmats = pose_matrix_fisher_sampling_torch(pose_U=pose_U,
                                                            pose_S=pose_S,
                                                            pose_V=pose_V,
                                                            num_samples=num_samples,
                                                            b=1.5,
                                                            oversampling_ratio=8)  # (1, num_samples, 23, 3, 3)
    if use_mean_shape:
        shape_to_use = shape_distribution.loc.expand(num_samples, -1)  # (num_samples, num_shape_params)
    else:
        shape_to_use = shape_distribution.sample([num_samples])[:, 0, :]  # (num_samples, num_shape_params) (batch_size = 1 is indexed out)
    smpl_samples = smpl_model(body_pose=pose_sample_rotmats[0, :, :, :],
                              global_orient=glob_rotmats.unsqueeze(1).expand(num_samples, -1, -1, -1),
                              betas=shape_to_use,
                              pose2rot=False)  # (num_samples, 6890, 3)
    vertices_samples = smpl_samples.vertices
    joints_samples = smpl_samples.joints

    mean_vertices = vertices_samples.mean(dim=0)
    avg_vertices_distance_from_mean = torch.norm(vertices_samples - mean_vertices, dim=-1).mean(dim=0)  # (6890,)

    return avg_vertices_distance_from_mean, vertices_samples, joints_samples


def joints2D_error_sorted_verts_sampling(pred_vertices_samples,
                                         pred_joints_samples,
                                         input_joints2D_heatmaps,
                                         pred_cam_wp):
    """
    Sort 3D vertex mesh samples according to consistency (error) between projected 2D joint samples
    and input 2D joints.
    :param pred_vertices_samples: (N, 6890, 3) tensor of candidate vertex mesh samples.
    :param pred_joints_samples: (N, 90, 3) tensor of candidate J3D samples.
    :param input_joints2D_heatmaps: (1, 17, img_wh, img_wh) tensor of 2D joint locations and confidences.
    :param pred_cam_wp: (1, 3) array with predicted weak-perspective camera.
    :return: pred_vertices_samples_error_sorted: (N, 6890, 3) tensor of J2D-error-sorted vertex mesh samples.
    """
    # Project 3D joint samples to 2D (using COCO joints)
    pred_joints_coco_samples = pred_joints_samples[:, ALL_JOINTS_TO_COCO_MAP, :]
    pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                    axes=torch.tensor([1., 0., 0.], device=pred_vertices_samples.device).float(),
                                                                    angles=np.pi,
                                                                    translations=torch.zeros(3, device=pred_vertices_samples.device).float())
    pred_joints2D_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)
    pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples,
                                                             input_joints2D_heatmaps.shape[-1])

    # Convert input 2D joint heatmaps into coordinates
    input_joints2D_coco, input_joints2D_coco_vis = convert_heatmaps_to_2Djoints_coordinates_torch(joints2D_heatmaps=input_joints2D_heatmaps,
                                                                                                  eps=1e-6)  # (1, 17, 2) and (1, 17)

    # Gather visible 2D joint samples and input
    pred_visible_joints2D_coco_samples = pred_joints2D_coco_samples[:, input_joints2D_coco_vis[0], :]  # (N, num vis joints, 2)
    input_visible_joints2D_coco = input_joints2D_coco[:, input_joints2D_coco_vis[0], :]  # (1, num vis joints, 2)

    # Compare 2D joint samples and input using Euclidean distance on image plane.
    j2d_l2es = torch.norm(pred_visible_joints2D_coco_samples - input_visible_joints2D_coco, dim=-1)  # (N, num vis joints)
    j2d_l2e_max, _ = torch.max(j2d_l2es, dim=-1)  # (N,)  # Max joint L2 error for each sample
    _, error_sort_idx = torch.sort(j2d_l2e_max, descending=False)

    pred_vertices_samples_error_sorted = pred_vertices_samples[error_sort_idx]

    return pred_vertices_samples_error_sorted
