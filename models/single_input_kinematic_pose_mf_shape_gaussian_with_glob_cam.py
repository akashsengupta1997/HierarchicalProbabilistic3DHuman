from collections import defaultdict
import torch
from torch import nn as nn
from torch.distributions import Normal

from models.resnet import resnet18, resnet50

from utils.rigid_transform_utils import rotmat_to_rot6d


def immediate_parents_to_all_parents(immediate_parents):
    parents_dict = defaultdict(list)
    for i in range(1, len(immediate_parents)):
        joint = i - 1
        immediate_parent = immediate_parents[i] - 1
        if immediate_parent >= 0:
            parents_dict[joint] += [immediate_parent] + parents_dict[immediate_parent]
    return parents_dict


class SingleInputKinematicPoseMFShapeGaussianwithGlobCam(nn.Module):
    def __init__(self,
                 smpl_parents,
                 resnet_in_channels=3,
                 resnet_layers=50,
                 use_6d_pose_for_glob=True,
                 add_delta_identity_to_pose_F=False,
                 pose_F_delta_identity_weight=1.,
                 feats_shape_glob_cam_embed_dim=64,
                 num_smpl_betas=10):
        """
        Input --> ResNet --> image features --> FC layers --> Hierarchical Kinematic Matrix Fisher over pose and Diagonal Gaussian over shape.
        Also get cam and glob separately to Gaussian distribution predictor.
        Pose predictions follow the kinematic chain.
        """
        super(SingleInputKinematicPoseMFShapeGaussianwithGlobCam, self).__init__()

        self.add_delta_identity_to_pose_F = add_delta_identity_to_pose_F
        self.pose_F_delta_identity_weight = pose_F_delta_identity_weight

        # Num pose parameters + Kinematic tree
        self.parents_dict = immediate_parents_to_all_parents(smpl_parents)
        print(self.parents_dict)
        self.num_joints = len(self.parents_dict)
        self.num_pose_params = self.num_joints * 3 * 3  # 3x3 matrix parameter for MF distribution for each joint.

        # Number of shape, glob and cam parameters
        self.num_shape_params = num_smpl_betas
        if use_6d_pose_for_glob:
            self.num_glob_params = 6
            init_glob = rotmat_to_rot6d(torch.eye(3)[None, :].float())
        else:
            self.num_glob_params = 3
            init_glob = torch.tensor([0.0, 0.0, 0.0]).float()
        self.register_buffer('init_glob', init_glob)
        self.num_cam_params = 3
        init_cam = torch.tensor([0.9, 0.0, 0.0]).float()  # Initialise orthographic camera scale at 0.9
        self.register_buffer('init_cam', init_cam)

        # ResNet Image Encoder
        if resnet_layers == 18:
            self.image_encoder = resnet18(in_channels=resnet_in_channels,
                                          pretrained=False)
            num_image_features = 512
            fc1_dim = 512
        elif resnet_layers == 50:
            self.image_encoder = resnet50(in_channels=resnet_in_channels,
                                          pretrained=False)
            num_image_features = 2048
            fc1_dim = 1024

        # FC Shape/Glob/Cam networks
        self.activation = nn.ELU()

        self.fc1 = nn.Linear(num_image_features, fc1_dim)

        self.fc_shape = nn.Linear(fc1_dim, self.num_shape_params * 2)  # Means and variances for SMPL betas and/or measurements
        self.fc_glob = nn.Linear(fc1_dim, self.num_glob_params)
        self.fc_cam = nn.Linear(fc1_dim, self.num_cam_params)

        self.fc_embed = nn.Linear(num_image_features + self.num_shape_params * 2 + self.num_glob_params + self.num_cam_params,
                                  feats_shape_glob_cam_embed_dim)

        # FC Pose networks for each joint
        self.fc_pose = nn.ModuleList()
        for joint in range(self.num_joints):
            num_parents = len(self.parents_dict[joint])
            input_dim = feats_shape_glob_cam_embed_dim + num_parents * (9 + 3 + 9)  # (passing (U, S, UV.T) for each parent to fc_pose - these have shapes (3x3), (3,), (3x3)
            self.fc_pose.append(nn.Sequential(nn.Linear(input_dim, feats_shape_glob_cam_embed_dim // 2),
                                              self.activation,
                                              nn.Linear(feats_shape_glob_cam_embed_dim // 2, 9)))

    def forward(self, input, input_feats=None):
        """
        input: (B, C, D, D) where B is batch size, C is number of channels and
        D is height and width.
        """
        if input_feats is None:
            input_feats = self.image_encoder(input)
        batch_size = input_feats.shape[0]
        device = input_feats.device

        x = self.activation(self.fc1(input_feats))

        # Shape
        shape_params = self.fc_shape(x)  # (bsize, num_smpl_betas * 2)
        shape_mean = shape_params[:, :self.num_shape_params]
        shape_log_std = shape_params[:, self.num_shape_params:]
        shape_dist = Normal(loc=shape_mean, scale=torch.exp(shape_log_std))

        # Glob rot and WP Cam
        delta_cam = self.fc_cam(x)
        delta_glob = self.fc_glob(x)
        glob = delta_glob + self.init_glob  # (bsize, num glob)
        cam = delta_cam + self.init_cam  # (bsize, 3)

        # Input Feats/Shape/Glob/Cam embed
        embed = self.activation(self.fc_embed(torch.cat([input_feats, shape_params, glob, cam], dim=1)))  # (bsize, feats_shape_glob_cam_embed_dim)

        # Pose
        pose_F = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)  # (bsize, 23, 3, 3)
        pose_U = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)  # (bsize, 23, 3, 3)
        pose_S = torch.zeros(batch_size, self.num_joints, 3, device=device)  # (bsize, 23, 3)
        pose_V = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)  # (bsize, 23, 3, 3)
        pose_U_proper = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)  # (bsize, 23, 3, 3)
        pose_S_proper = torch.zeros(batch_size, self.num_joints, 3, device=device)  # (bsize, 23, 3)
        # pose_V_proper = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)  # (bsize, 23, 3, 3)
        pose_rotmats_mode = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)  # (bsize, 23, 3, 3)

        for joint in range(self.num_joints):
            parents = self.parents_dict[joint]
            fc_joint = self.fc_pose[joint]

            if len(parents) > 0:
                parents_U_proper = pose_U_proper[:, parents, :, :].view(batch_size, -1)  # (bsize, num parents * 3 * 3)
                parents_S_proper = pose_S_proper[:, parents, :].view(batch_size, -1)  # (bsize, num parents * 3)
                parents_mode = pose_rotmats_mode[:, parents, :, :].view(batch_size, -1)  # (bsize, num parents * 3 * 3)

                joint_F = fc_joint(torch.cat([embed, parents_U_proper, parents_S_proper, parents_mode], dim=1)).view(-1, 3, 3)  # (bsize, 3, 3)
            else:
                joint_F = fc_joint(embed).view(-1, 3, 3)  # (bsize, 3, 3)

            if self.add_delta_identity_to_pose_F:
                joint_F = joint_F + self.pose_F_delta_identity_weight * torch.eye(3, device=device)[None, :, :].expand_as(joint_F)

            joint_U, joint_S, joint_V = torch.svd(joint_F.cpu())  # (bsize, 3, 3), (bsize, 3), (bsize, 3, 3)
            with torch.no_grad():
                det_joint_U, det_joint_V = torch.det(joint_U).to(device), torch.det(joint_V).to(device)  # (bsize,), (bsize,)
            joint_U, joint_S, joint_V = joint_U.to(device), joint_S.to(device), joint_V.to(device)

            joint_U_proper = joint_U.clone()
            joint_S_proper = joint_S.clone()
            joint_V_proper = joint_V.clone()
            joint_U_proper[:, :, 2] *= det_joint_U.unsqueeze(-1)
            joint_S_proper[:, 2] *= det_joint_U * det_joint_V
            joint_V_proper[:, :, 2] *= det_joint_V.unsqueeze(-1)

            joint_rotmat_mode = torch.matmul(joint_U_proper, joint_V_proper.transpose(dim0=-1, dim1=-2))

            pose_F[:, joint, :, :] = joint_F
            pose_U[:, joint, :, :] = joint_U
            pose_S[:, joint, :] = joint_S
            pose_V[:, joint, :, :] = joint_V
            pose_U_proper[:, joint, :, :] = joint_U_proper
            pose_S_proper[:, joint, :] = joint_S_proper
            # pose_V_proper[:, joint, :, :] = joint_V_proper
            pose_rotmats_mode[:, joint, :, :] = joint_rotmat_mode

        # print((torch.det(pose_rotmats_mode) - 1).abs().max(), (torch.det(pose_U_proper) - 1).abs().max())
        return pose_F, pose_U, pose_S, pose_V, pose_rotmats_mode, shape_dist, glob, cam

