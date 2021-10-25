import torch
import numpy as np
from smplx import SMPL as _SMPL
try:
    from smplx.body_models import ModelOutput as SMPLOutput
except ImportError:
    from smplx.utils import SMPLOutput
from smplx.lbs import vertices2joints

from configs import paths


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load(paths.J_REGRESSOR_EXTRA)
        J_regressor_cocoplus = np.load(paths.COCOPLUS_REGRESSOR)
        J_regressor_h36m = np.load(paths.H36M_REGRESSOR)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra,
                                                               dtype=torch.float32))
        self.register_buffer('J_regressor_cocoplus', torch.tensor(J_regressor_cocoplus,
                                                                  dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m,
                                                              dtype=torch.float32))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        cocoplus_joints = vertices2joints(self.J_regressor_cocoplus, smpl_output.vertices)
        h36m_joints = vertices2joints(self.J_regressor_h36m, smpl_output.vertices)
        all_joints = torch.cat([smpl_output.joints, extra_joints, cocoplus_joints,
                                h36m_joints], dim=1)
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=all_joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output
