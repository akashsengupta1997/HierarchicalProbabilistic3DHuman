import cv2
import torch
import numpy as np
from torch.nn import functional as F
try:
    from pytorch3d.transforms.so3 import so3_log_map, so3_exponential_map
except ImportError:
    print('Failed to import pytorch3d in rigid_transform_utils.py')


def aa_rotate_rotmats_pytorch3d(rotmats, axes, angles, rot_mult_order='post'):
    """
    Batched rotation of rotation matrices about given axes and angles.

    :param rotmats: (B, 3, 3), batch of rotation matrices
    :param axes: (B, 3) or (3,), rotation axes (may be batched)
    :param angles: (B, 1) or scalar, rotation angles in radians (may be batched)
    :return: rotated_rotvecs (B, 3) and rotated_rotmats (B, 3, 3)
    """
    assert rot_mult_order in ['pre', 'post']
    r = axes * angles
    if r.dim() < 2:
        r = r[None, :].expand(rotmats.shape[0], -1)
    R = so3_exponential_map(log_rot=r)  # (B, 3, 3)
    if rot_mult_order == 'post':
        rotated_rotmats = torch.matmul(rotmats, R)
    elif rot_mult_order == 'pre':
        rotated_rotmats = torch.matmul(R, rotmats)
    rotated_rotvecs = so3_log_map(R=rotated_rotmats)

    return rotated_rotvecs, rotated_rotmats


def aa_rotate_rotmats(axis, angle, rotmats, rot_mult_order='post'):
    """
    This does the same thing as aa_rotate_rotmats_pytorch3d, except using openCV instead of pytorch3d.
    This is preferred when computing rotated rotation vectors (SO(3) log map) because pytorch3d's
    SO(3) log map is broken for R = I.
    However pytorch3d function is batched and only requires torch, so should be faster - use when
    rotation vectors are not needed (e.g. during training).

    :param rotmats: (B, 3, 3), batch of rotation matrices
    :param axis: (3, ) numpy array, axis of rotation
    :param angle: scalar, angle of rotation
    :return: rotated_vecs (B, 3) and rotated_rotmats (B, 3, 3)
    """
    assert rot_mult_order in ['pre', 'post']
    R = cv2.Rodrigues(np.array(axis)*angle)[0]
    rotmats = rotmats.cpu().detach().numpy()
    if rot_mult_order == 'post':
        rotated_rotmats = np.matmul(rotmats, R)
    elif rot_mult_order == 'pre':
        rotated_rotmats = np.matmul(R, rotmats)
    rotated_vecs = []
    for i in range(rotated_rotmats.shape[0]):
        rotated_vecs.append(cv2.Rodrigues(rotated_rotmats[i, :, :])[0].squeeze())
    rotated_vecs = torch.from_numpy(np.stack(rotated_vecs, axis=0)).float()
    return rotated_vecs, torch.from_numpy(rotated_rotmats).float()


def aa_rotate_translate_points_pytorch3d(points, axes, angles, translations):
    """
    Rotates and translates batch of points from a mesh about given axes and angles.
    :param points: B, N, 3, batch of meshes with N points each
    :param axes: (B,3) or (3,), rotation axes
    :param angles: (B,1) or scalar, rotation angles in radians
    :param translations: (B,3) or (3,), translation vectors
    :return:
    """
    r = axes * angles
    if r.dim() < 2:
        r = r[None, :].expand(points.shape[0], -1)
    R = so3_exponential_map(log_rot=r)  # (B, 3, 3)
    points = torch.einsum('bij,bkj->bki', R, points)
    points = points + translations

    return points


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) or (B, 24*6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)  # Ensuring columns are unit vectors
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)  # Ensuring column 1 and column 2 are orthogonal with Gram-Schmidt orthogonalisation
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(R, stack_columns=False):
    """
    :param R: (B, 3, 3)
    :param stack_columns:
        if True, 6D pose representation is [1st col of R, 2nd col of R]^T = [R11, R21, R31, R12, R22, R32]^T
        if False, 6D pose representation is [R11, R12, R21, R22, R31, R32]^T
        Set to False if doing inverse of rot6d_to_rotmat
    :return: rot6d: (B, 6)
    """
    if stack_columns:
        rot6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)
    else:
        rot6d = R[:, :, :2].contiguous().view(-1, 6)
    return rot6d


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def quat_to_rotmat_numpy(quat):
    """
    :param quat: (B, 4)
    :return:
    """
    norm_quat = quat
    norm_quat = norm_quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.shape[0]

    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = np.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                       2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                       2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], axis=1).reshape(B, 3, 3)
    return rotMat



def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


