import os
import cv2
import numpy as np
import torch
import pickle
import argparse


from configs import paths
from utils.cam_utils import perspective_project_torch
from models.smpl_official import SMPL


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height,
                            do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans


def get_single_image_crop(image, bbox, scale=1.2, crop_size=224):
    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size,
        patch_height=crop_size,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    return crop_image


def pw3d_eval_extract(dataset_path, out_path, crop_wh=512):
    bbox_scale_factor = 1.2

    smpl_male = SMPL(paths.SMPL, batch_size=1, gender='male').to(device)
    smpl_female = SMPL(paths.SMPL, batch_size=1, gender='female').to(device)

    # imgnames_, scales_, centers_, parts_ = [], [], [], []
    cropped_frame_fnames_, whs_, centers_, = [], [], []
    poses_, shapes_, genders_ = [], [], []

    sequence_files = sorted([os.path.join(dataset_path, 'sequenceFiles', 'test', f)
                             for f in os.listdir(os.path.join(dataset_path, 'sequenceFiles', 'test'))
                             if f.endswith('.pkl')])

    for filename in sequence_files:
        print('\n\n\n', filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        smpl_poses = data['poses']  # list of (num frames, 72) pose params for each person
        smpl_betas = data['betas']  # list of (10,) or (300,) shape params for each person
        poses2d = data['poses2d']  # list of (num frames, 3, 18) 2d kps for each person
        cam_extrinsics = data['cam_poses']  # array of (num frames, 4, 4) cam extrinsics
        cam_K = data['cam_intrinsics']  # array of (3, 3) cam intrinsics.
        genders = data['genders'] # list of genders for each person
        valid = data['campose_valid']  # list of (num frames,) boolean arrays for each person, indicating whether camera pose has been aligned to that person (for trans).
        trans = data['trans']  # list of (num frames, 3) translations in SMPL space for each person, to align them with image data (after projection)
        num_people = len(smpl_poses)  # Number of people in sequence
        num_frames = len(smpl_poses[0])  # Number of frames in sequence
        seq_name = str(data['sequence'])
        print('smpl poses', len(smpl_poses), smpl_poses[0].shape,
              'smpl betas', len(smpl_betas), smpl_betas[0].shape,
              'poses2d', len(poses2d), poses2d[0].shape,
              'global poses', cam_extrinsics.shape,
              'cam_K', cam_K.shape,
              'genders', genders, type(genders),
              'valid', len(valid), valid[0].shape, np.sum(valid[0]), np.sum(valid[-1]),
              'trans', len(trans), trans[0].shape,
              'num people', num_people, 'num frames', num_frames, 'seq name', seq_name, '\n')

        cam_K = torch.from_numpy(cam_K[None, :]).float().to(device)
        for person_num in range(num_people):
            # Get valid frames flags, shape and gender
            valid_frames = valid[person_num].astype(np.bool)
            shape = smpl_betas[person_num][:10]
            torch_shape = torch.from_numpy(shape[None, :]).float().to(device)
            gender = genders[person_num]

            for frame_num in range(num_frames):
                if valid_frames[frame_num]:  # Only proceed if frame has valid camera pose for person
                    # Get bounding box using projected vertices
                    pose = smpl_poses[person_num][frame_num]
                    cam_R = cam_extrinsics[frame_num][:3, :3]
                    cam_t = cam_extrinsics[frame_num][:3, 3]
                    frame_trans = trans[person_num][frame_num]
                    pose = torch.from_numpy(pose[None, :]).float().to(device)
                    cam_t = torch.from_numpy(cam_t[None, :]).float().to(device)
                    cam_R = torch.from_numpy(cam_R[None, :, :]).float().to(device)
                    frame_trans = torch.from_numpy(frame_trans[None, :]).float().to(device)
                    if gender == 'm':
                        smpl_out = smpl_male(body_pose=pose[:, 3:],
                                             global_orient=pose[:, :3],
                                             betas=torch_shape,
                                             transl=frame_trans)
                    elif gender == 'f':
                        smpl_out = smpl_female(body_pose=pose[:, 3:],
                                               global_orient=pose[:, :3],
                                               betas=torch_shape,
                                               transl=frame_trans)
                    vertices = smpl_out.vertices
                    projected_aligned_vertices = perspective_project_torch(vertices, cam_R,
                                                                           cam_t, cam_K=cam_K)
                    projected_aligned_vertices = projected_aligned_vertices[0].cpu().detach().numpy()
                    bbox = [min(projected_aligned_vertices[:, 0]),
                            min(projected_aligned_vertices[:, 1]),
                            max(projected_aligned_vertices[:, 0]),
                            max(projected_aligned_vertices[:, 1])]  # (x1, y1, x2, y2) where x is cols and y is rows from top right corner.
                    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    wh = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

                    # Save cropped frame using bounding box
                    image_fpath = os.path.join(dataset_path, 'imageFiles', seq_name,
                                               'image_{}.jpg'.format(str(frame_num).zfill(5)))
                    image = cv2.imread(image_fpath)
                    centre_wh_bbox = center + [wh, wh]
                    cropped_image = get_single_image_crop(image, centre_wh_bbox,
                                                          scale=bbox_scale_factor,
                                                          crop_size=crop_wh)
                    cropped_image_fname = seq_name + '_image_{}_person_{}.png'.format(str(frame_num).zfill(5),
                                                                                      str(person_num).zfill(3))
                    cropped_image_fpath = os.path.join(out_path, 'cropped_frames',
                                                       cropped_image_fname)
                    cv2.imwrite(cropped_image_fpath, cropped_image)

                    # Transform global using cam extrinsics pose before storing
                    pose = pose[0].cpu().detach().numpy()
                    cam_R = cam_R[0].cpu().detach().numpy()
                    pose[:3] = cv2.Rodrigues(np.dot(cam_R, cv2.Rodrigues(pose[:3])[0]))[0].T[0]

                    # Store everything in lists
                    cropped_frame_fnames_.append(cropped_image_fname)
                    centers_.append(center)
                    whs_.append(wh)
                    poses_.append(pose)
                    shapes_.append(shape)
                    genders_.append(gender)
                    # print(cropped_image_fname, shape.shape, pose.shape, center, wh, gender)

    # Store all data in npz file.
    out_file = os.path.join(out_path, '3dpw_test.npz')
    np.savez(out_file, imgname=cropped_frame_fnames_,
             center=centers_,
             wh=whs_,
             pose=poses_,
             shape=shapes_,
             gender=genders_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    out_path = os.path.join(args.dataset_path, 'test')
    if not os.path.isdir(out_path):
        os.makedirs(os.path.join(out_path, 'cropped_frames'))
    pw3d_eval_extract(args.dataset_path, out_path)

