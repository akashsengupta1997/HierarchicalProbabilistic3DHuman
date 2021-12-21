import shutil
import os
from tqdm import tqdm
import argparse


def copy_lsun_images_to_train_files_dir(lsun_dir, train_files_dir):

    train_save_path = os.path.join(train_files_dir, 'lsun_backgrounds', 'train')
    val_save_path = os.path.join(train_files_dir, 'lsun_backgrounds', 'val')
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)

    for subdir in sorted([f for f in os.listdir(lsun_dir)
                   if os.path.isdir(os.path.join(lsun_dir, f)) and 'images' in f]):
        if 'train' in subdir:
            split = 'train'
        elif 'val' in subdir:
            split = 'val'
        print('SUBDIR:', subdir, split)
        for image_fname in tqdm([f for f in os.listdir(os.path.join(lsun_dir, subdir)) if f.endswith('.jpg')]):
            # print("From:", os.path.join(lsun_dir, subdir, image_fname), " To:", os.path.join(otf_smpl_dir, 'lsun_backgrounds', split))
            shutil.copy(os.path.join(lsun_dir, subdir, image_fname),
                        os.path.join(train_files_dir, 'lsun_backgrounds', split))  # Change copy to move if you don't want to copy all the data for memory-saving purposes.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsun_dir', type=str)
    parser.add_argument('--train_files_dir', type=str)
    args = parser.parse_args()

    copy_lsun_images_to_train_files_dir(lsun_dir=args.lsun_dir,
                                        train_files_dir=args.train_files_dir)