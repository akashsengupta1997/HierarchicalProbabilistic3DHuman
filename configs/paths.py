# ------------------- SMPL Files -------------------
SMPL = './model_files/smpl'
J_REGRESSOR_EXTRA = './model_files/J_regressor_extra.npy'
COCOPLUS_REGRESSOR= './model_files/cocoplus_regressor.npy'
H36M_REGRESSOR = './model_files/J_regressor_h36m.npy'

# ------------------- DensePose Files for Textured Rendering -------------------
DP_UV_PROCESSED_FILE = './model_files/UV_Processed.mat'

# ------------------------- Eval Datasets -------------------------
PW3D_PATH = '/scratch2/as2562/datasets/3DPW/test'
SSP3D_PATH = '/scratch2/as2562/datasets/ssp_3d'

# ------------------------- Train Datasets -------------------------
TRAIN_POSES_PATH = '/scratch/as2562/datasets/on_the_fly_smpl/amass_up3d_3dpw_train.npz'
TRAIN_TEXTURES_PATH = '/scratch/as2562/datasets/on_the_fly_smpl/densepose_textures/surreal_mgn_textures_train.npz'
TRAIN_BACKGROUNDS_PATH = '/scratch/as2562/datasets/on_the_fly_smpl/lsun_backgrounds/train'
VAL_POSES_PATH = '/scratch/as2562/datasets/on_the_fly_smpl/up3d_3dpw_val.npz'
VAL_TEXTURES_PATH = '/scratch/as2562/datasets/on_the_fly_smpl/densepose_textures/surreal_mgn_textures_val.npz'
VAL_BACKGROUNDS_PATH = '/scratch/as2562/datasets/on_the_fly_smpl/lsun_backgrounds/val'
