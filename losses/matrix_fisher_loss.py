"""
Code adapted from https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions
See Equations 85-90 in https://arxiv.org/pdf/1710.03746.pdf for more details.
"""

import torch
import torch.nn as nn

# Bessel function polynomial approximation coefficients from https://omlc.org/software/mc/conv-src/convnr.c
bessel0_exp_scaled_coeffs_a = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
bessel0_exp_scaled_coeffs_b = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]


def horners_method(coeffs,
                   x):
    """
    Horner's method of evaluating a polynomial with coefficients given by coeffs and
    input x.
    z = arr[0] + arr[1]x + arr[2]x^2 + ... + arr[n]x^n
    :param coeffs: List/Tensor of coefficients of polynomial (in standard form)
    :param x: Tensor of input values.
    :return z: Tensor of output values, same shape as x.
    """
    z = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(coeffs[0])
    for i in range(1, len(coeffs)):
        z.mul_(x).add_(coeffs[i])
    return z


def bessel0_exp_scaled(x):
    """
    Approximates the exponentially-scaled modified bessel function of the first kind
    I_0_bar(x) =  I_0(x) / exp(|x|) (https://arxiv.org/pdf/1710.03746.pdf Appendix C).
    Exponential division (scaling) is for numerical stability since I_0(x) grows
    very quickly with x.
    https://omlc.org/software/mc/conv-src/convnr.c
    :param x: Tensor of input values.
    :return I_0_bar: Tensor of output values, same shape as x.
    """
    abs_x = torch.abs(x)
    mask = abs_x <= 3.75
    I_0_bar_a = horners_method(bessel0_exp_scaled_coeffs_a, (abs_x / 3.75) ** 2) / torch.exp(abs_x)
    I_0_bar = horners_method(bessel0_exp_scaled_coeffs_b, 3.75 / abs_x) / torch.sqrt(abs_x)
    I_0_bar[mask] = I_0_bar_a[mask]
    return I_0_bar


def torch_trapezoid_integral(func,
                             func_args,
                             from_x,
                             to_x,
                             num_traps):
    """
    Integrates func from from_x to to_x using the trapezoid rule, with num_traps - 1 trapezoids.
    :param func: Function to integrate (the integrand)
    :param func_args: Arguments of f (but not the variable being integrated over)
        In practice this is proper singular values s0, s1, s2 with shape (B, 3)
    :param from_x lower limit of integration
    :param to_x upper limit of integration
    :param num_traps: number of trapezoids + 1
    :return integral of func from from_x to to_x
    """
    with torch.no_grad():
        range = torch.arange(num_traps, dtype=func_args.dtype, device=func_args.device)
        x = (range * ((to_x-from_x) / (num_traps - 1)) + from_x).view(1, num_traps)  # (x values: [from_x, to_x])
        weights = torch.empty((1, num_traps), dtype=func_args.dtype, device=func_args.device).fill_(1)
        weights[0, 0] = 1/2
        weights[0, -1] = 1/2
        y = func(x, func_args)
        return torch.sum(y * weights, dim=1) * (to_x - from_x)/(num_traps - 1)


def integrand_normconst_forward_exp_scaled(u, s):
    """
    Integrand required to compute exp scaled normalising constant c_bar(S) = c(S) / exp(tr(S)).

    Computed using modified Bessel functions of the first kind (I_0).

    See Eqn 86 of https://arxiv.org/pdf/1710.03746.pdf for more details.

    :param u: (1, N) Tensor of input values (will be integrated over).
    :param s: (B, 3) Tensor of batch of proper singular values ordered from big to small.
               Will be using s_i, s_j, s_k = s_2, s_3, s_1
    :return integrand_c_bar: (B, N) Tensor of batch of integrand values over given x.
    """
    # s is sorted from big to small
    factor1 = (s[:, [1]] - s[:, [2]]) * 0.5 * (1 - u)
    factor1 = bessel0_exp_scaled(factor1)

    factor2 = (s[:, [1]] + s[:, [2]]) * 0.5 * (1 + u)
    factor2 = bessel0_exp_scaled(factor2)

    factor3 = torch.exp((s[:, [2]] + s[:, [0]]) * (u - 1))

    integrand_c_bar = factor1 * factor2 * factor3
    return integrand_c_bar


def integrand_dlognormconst_ds_backward(u, s):
    """
    Integrand required to compute derivative of log norm constant dlog(c(S)) / ds_k.

    Since log(c(S)) = log(c_bar(S)) + tr(S),
    dlog(c(S)) / ds_k = 1/c_bar(S) dc_bar(S) / ds_k + dtr(S) / ds_k
    Since dtr(S) / ds_k = 1,
    dlog(c(S)) / ds_k = 1/c_bar(S) dc_bar(S) / ds_k + 1
                      = 1/c_bar(S) (dc_bar(S) / ds_k + c_bar(S))
    The integrand computed here is for dc_bar(S) / ds_k + c_bar(S).

    See Eqn. 85-90 of https://arxiv.org/pdf/1710.03746.pdf for more details.

    :param u: (1, N) Tensor of input values (will be integrated over).
    :param s: (B, 3) Tensor of batch of proper singular values.
               Not necessarily ordered big to small (1, 2, 3), but in circular shifts
               (1, 2, 3), (2, 3, 1), (3, 1, 2).
    :return: integrand_dlogc_dcs_k: (B, N
    """
    s_i = torch.max(s[:, 1:], dim=1, keepdim=True).values
    s_j = torch.min(s[:, 1:], dim=1, keepdim=True).values
    s_k = s[:, [0]]

    factor1 = (s_i - s_j) * 0.5 * (1 - u)
    factor1 = bessel0_exp_scaled(factor1)

    factor2 = (s_i + s_j) * 0.5 * (1 + u)
    factor2 = bessel0_exp_scaled(factor2)

    factor3 = torch.exp((s_j + s_k) * (u - 1))

    integrand_dlogc_dcs_k = factor1 * factor2 * factor3 * u  # Don't have u-1 here because this is integrand of dc_bar(S)/ds_k + c_bar(S).
    return integrand_dlogc_dcs_k


class LogMFNormConstant(torch.autograd.Function):
    """
    Pytorch Autograd function with:
        output: log normalising constant log(c(S)) = log(c_bar(S)) + tr(S)
        input: proper singular values S = s0, s1, s2 (which will be predicted using NN).

    Backward method gives gradient of output w.r.t. input ie:
        dlog(c(S)) / ds_k for singular values s0, s1, s2.
        dlog(c(S)) / ds_k = 1/c_bar(S) dc_bar(S) / ds_k + dtr(S) / ds_k
        Since dtr(S) / ds_k = 1,
        dlog(c(S))/ds_k = 1/c_bar(S) (dc_bar(S) / ds_k + c_bar(S))

    See Eqn. 85-90 of https://arxiv.org/pdf/1710.03746.pdf for more details.
    """
    @staticmethod
    def forward(ctx, S):
        """
        :param ctx: Pytorch context object, use for storing things required in the backward pass.
        :param S: (B, 3) tensor, batch of proper singular values, ordered big to small.
                      In practice, B = batch_size * num_smpl_joints
        :return: log_c: (B,) tensor, batch of log norm constants corresponding to
                        singular values in input.
        """
        num_traps = 512  # Number of trapezoids + 1 for integral

        c_bar = 0.5 * torch_trapezoid_integral(func=integrand_normconst_forward_exp_scaled,
                                               func_args=S,
                                               from_x=-1,
                                               to_x=1,
                                               num_traps=num_traps)  # c_bar(S), shape is (B,)
        ctx.save_for_backward(S, c_bar)  # Save for gradient computation in backward pass

        log_c_bar = torch.log(c_bar)  # log(c_bar(S))
        log_trace_S = torch.sum(S, dim=1)  # tr(S)

        log_c = log_c_bar + log_trace_S  # log(c(S)) = log(c_bar(S)) + tr(S)
        return log_c

    @staticmethod
    def backward(ctx, grad_log_c):
        """
        :param ctx: Pytorch context object, use for storing things required in the backward pass.
        :param grad_log_c: (B,) tensor, gradient of loss w.r.t log c(S).
        :return: grad_singvals: (B, 3) tensor, gradient of loss w.r.t S (i.e. singular values s0, s1, s2).
        """
        S, c_bar = ctx.saved_tensors  # S is proper singular values, c_bar is exp scaled log norm constant.
        num_traps = 512  # Number of trapezoids + 1 for integral

        dc_bar_dS = torch.empty((S.shape[0], 3), dtype=S.dtype, device=S.device)
        for i in range(3):
            S_shifted = torch.cat((S[:, i:], S[:, :i]), dim=1)  # Cyclic shifts of singular values
            dc_bar_dS[:, i] = 0.5 * torch_trapezoid_integral(func=integrand_dlognormconst_ds_backward,
                                                             func_args=S_shifted,
                                                             from_x=-1,
                                                             to_x=1,
                                                             num_traps=num_traps)  # dc_bar(S) / ds_k + c_bar(S)
        dlogc_dS = dc_bar_dS / c_bar.view(-1, 1)
        grad_S = dlogc_dS * grad_log_c.view(-1, 1)
        return grad_S.view(-1, 3)


def matrix_fisher_nll(pred_F,
                      pred_U,
                      pred_S,
                      pred_V,
                      target_R,
                      overreg=1.025):
    """
    Outputs NLL of target rotation matrices target_R, under matrix fisher distribution with
    predicted matrix parameter pred_F.
    If torch.svd fails, returns zero loss, unless it fails consistently then ends training run?
    :param pred_F: (*, 3, 3)
    :param target_R: (*, 3, 3)
    :param pred_U: (*, 3, 3) from SVD of F
    :param pred_S: (*, 3) from SVD of F
    :param pred_V: (*, 3, 3) from SVD of F
    :param overreg: amount of over-regularisation: multplicative scaling applied to log normalising constant.
                    log normalising constant acts as a regularising term.
    :returns NLL loss: (*,)
    """
    if pred_F.dim() > 3:  # Multiple batch dimensions - will be the case in practice since 23 SMPL rotations per sample
        pred_F = pred_F.view(-1, 3, 3)  # (N, 3, 3) where N is product of batch dimensions.
        pred_U = pred_U.view(-1, 3, 3)
        pred_S = pred_S.view(-1, 3)
        pred_V = pred_V.view(-1, 3, 3)
        target_R = target_R.contiguous().view(-1, 3, 3)

    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(pred_U, pred_V.transpose(1, 2)).detach().cpu()).to(pred_S.device)  # det(UV) = 1 or -1 depending if det(U) == det(V) or not
    pred_S_proper = pred_S.clone()  # (N, 3)
    pred_S_proper[:, 2] *= s3sign  # Proper singular values: s3 = s3 * det(UV)

    log_norm_constant = LogMFNormConstant.apply(pred_S_proper)  # log(c(S)) = log(c_bar(S)) + tr(S), shape is (N,)
    log_exponent = -torch.matmul(pred_F.view(-1, 1, 9), target_R.view(-1, 9, 1)).view(-1)  # -tr(A^T R), shape is (N,)
    return log_exponent + overreg * log_norm_constant  # NLL, shape is (N,) where N is product of batch dimensions.


class PoseMFShapeGaussianLoss(nn.Module):
    """
    NLL for Matrix-Fisher distribution over SMPL pose rotation matrices +
    NLL for Gaussian distribution over SMPL shape parameters +
    MSE loss for joints2D and glob rotmats.
    Optional MSE losses for verts and joints3D.
    """
    def __init__(self,
                 loss_config,
                 img_wh):
        super(PoseMFShapeGaussianLoss, self).__init__()

        self.loss_config = loss_config

        self.img_wh = img_wh
        self.joints2D_loss = nn.MSELoss(reduction=loss_config.REDUCTION)
        self.glob_rotmats_loss = nn.MSELoss(reduction=loss_config.REDUCTION)
        self.verts3D_loss = nn.MSELoss(reduction=loss_config.REDUCTION)
        self.joints3D_loss = nn.MSELoss(reduction=loss_config.REDUCTION)

    def forward(self, target_dict, pred_dict):

        # Pose NLL
        pose_nll = matrix_fisher_nll(pred_F=pred_dict['pose_params_F'],
                                     pred_U=pred_dict['pose_params_U'],
                                     pred_S=pred_dict['pose_params_S'],
                                     pred_V=pred_dict['pose_params_V'],
                                     target_R=target_dict['pose_params_rotmats'],
                                     overreg=self.loss_config.MF_OVERREG)
        if self.loss_config.REDUCTION == 'mean':
            pose_nll = torch.mean(pose_nll)
        elif self.loss_config.REDUCTION == 'sum':
            pose_nll = torch.sum(pose_nll)

        # Shape NLL
        shape_nll = -(pred_dict['shape_params'].log_prob(target_dict['shape_params']).sum(dim=1))  # (batch_size,)
        if self.loss_config.REDUCTION == 'mean':
            shape_nll = torch.mean(shape_nll)
        elif self.loss_config.REDUCTION == 'sum':
            shape_nll = torch.sum(shape_nll)

        # Joints2D MSE
        target_joints2D = target_dict['joints2D']
        target_joints2D_vis = target_dict['joints2D_vis']
        pred_joints2D = pred_dict['joints2D']

        target_joints2D = target_joints2D[:, None, :, :].expand_as(pred_joints2D)  # Selecting visible 2D joint targets and predictions
        target_joints2D_vis = target_joints2D_vis[:, None, :].expand(-1, pred_joints2D.shape[1], -1)
        pred_joints2D = pred_joints2D[target_joints2D_vis, :]
        target_joints2D = target_joints2D[target_joints2D_vis, :]

        target_joints2D = (2.0 * target_joints2D) / self.img_wh - 1.0  # normalising joints to [-1, 1] range.
        joints2D_loss = self.joints2D_loss(pred_joints2D, target_joints2D)

        # Glob Rotmats MSE
        glob_rotmats_loss = self.glob_rotmats_loss(pred_dict['glob_rotmats'], target_dict['glob_rotmats'])

        # Verts3D MSE
        verts_loss = self.verts3D_loss(pred_dict['verts'], target_dict['verts'])

        # Joints3D MSE
        joints3D_loss = self.joints3D_loss(pred_dict['joints3D'], target_dict['joints3D'])

        total_loss = pose_nll * self.loss_config.WEIGHTS.POSE \
                     + shape_nll * self.loss_config.WEIGHTS.SHAPE \
                     + joints2D_loss * self.loss_config.WEIGHTS.JOINTS2D \
                     + glob_rotmats_loss * self.loss_config.WEIGHTS.GLOB_ROTMATS \
                     + verts_loss * self.loss_config.WEIGHTS.VERTS3D \
                     + joints3D_loss * self.loss_config.WEIGHTS.JOINTS3D

        return total_loss

