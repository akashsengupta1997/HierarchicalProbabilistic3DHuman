# Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild
Akash Sengupta, Ignas Budvytis, Roberto Cipolla  
ICCV 2021  
[[paper+supplementary](https://arxiv.org/pdf/2110.00990.pdf)][[poster](https://www.youtube.com/watch?v=w7k9UC3sfGA)][[results video](https://www.youtube.com/watch?v=qVrvOebDBs4)]

This is the official code repository of the above paper, which takes a probabilistic approach to 3D human shape and pose estimation and predicts multiple plausible 3D reconstruction samples given an input image. 

![teaser](teaser.gif)

This repository contains inference, training (TODO) and evaluation code. A few weaknesses of this approach, and future research directions, are listed below (TODO).
If you find this code useful in your research, please cite the following publication:
```
@InProceedings{sengupta2021hierprobhuman,
               author = {Sengupta, Akash and Budvytis, Ignas and Cipolla, Roberto},
               title = {{Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild}},
               booktitle = {International Conference on Computer Vision},
               month = {October},
               year = {2021}                         
}
```

## Installation

### Requirements
- Linux or macOS
- Python ≥ 3.6

### Instructions
We recommend using a virtual environment to install relevant dependencies:
```
python3 -m venv HierProbHuman
source HierProbHuman/bin/activate
```
Install torch and torchvision (the code has been tested with v1.6.0 of torch), as well as other dependencies: 
```
pip install torch==1.6.0 torchvision==0.7.0
pip install -r requirements.txt
``` 
Finally, install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/INSTALL.md), which we use for data generation during training and visualisation during inference. To do so, you will need to first install the CUB library following the instructions [here](https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/INSTALL.md). Then you may install pytorch3d - note that the code has been tested with v0.3.0 of pytorch3d, and we recommend installing this version using: 
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.3.0"
```

### Model files
You will need to download the SMPL model. The [neutral model](http://smplify.is.tue.mpg.de) is required for training and running the demo code. If you want to evaluate the model on datasets with gendered SMPL labels (such as 3DPW and SSP-3D), the male and female models are available [here](http://smpl.is.tue.mpg.de). You will need to convert the SMPL model files to be compatible with python3 by removing any chumpy objects. To do so, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

Download pre-trained model checkpoints for our 3D Shape/Pose network, as well as for 2D Pose [HRNet-W48](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) from [here](https://drive.google.com/drive/folders/1WHdbAaPM8-FpnwMuCdVEchskgKab3gel?usp=sharing). 

Place the SMPL model files and network checkpoints in the `model_files` directory, which should have the following structure. If the files are placed elsewhere, you will need to update `configs/paths.py` accordingly.

    HierarchicalProbabilistic3DHuman
    ├── model_files                                  # Folder with model files
    │   ├── smpl
    │   │   ├── SMPL_NEUTRAL.pkl                     # Gender-neutral SMPL model
    │   │   ├── SMPL_MALE.pkl                        # Male SMPL model
    │   │   ├── SMPL_FEMALE.pkl                      # Female SMPL model
    │   ├── poseMF_shapeGaussian_net_weights.tar     # Pose/Shape distribution predictor checkpoint
    │   ├── pose_hrnet_w48_384x288.pth               # Pose2D HRNet checkpoint
    │   ├── cocoplus_regressor.npy                   # Cocoplus joints regressor
    │   ├── J_regressor_h36m.npy                     # Human3.6M joints regressor
    │   ├── J_regressor_extra.npy                    # Extra joints regressor
    │   └── UV_Processed.mat                         # DensePose UV coordinates for SMPL mesh             
    └── ...
 
## Inference
`run_predict.py` is used to run inference on a given folder of input images. For example, to run inference on the demo folder, do:
```
python run_predict.py --image_dir ./demo/ --save_dir ./output/ --visualise_samples --visualise_uncropped
```
This will first detect human bounding boxes in the input images using Mask-RCNN. If your input images are already cropped and centred around the subject of interest, you may skip this step using `--cropped_images` as an option. The 3D Shape/Pose network is somewhat sensitive to cropping and centering - this is a good place to start troubleshooting in case of poor results.

Inference can be slow due to the rejection sampling procedure used to estimate per-vertex 3D uncertainty. If you are not interested in per-vertex uncertainty, you may modify `predict/predict_poseMF_shapeGaussian_net.py` by commenting out code related to sampling, and use a plain texture to render meshes for visualisation (this will be cleaned up and added as an option to in the `run_predict.py` future).

## Evaluation
`run_evaluate.py` is used to evaluate our method on the 3DPW and SSP-3D datasets. A description of the metrics used to measure performance is given in `metrics/eval_metrics_tracker.py`.

Download SSP-3D from [here](https://github.com/akashsengupta1997/SSP-3D). Update `configs/paths.py` with the path pointing to the un-zipped SSP-3D directory. Evaluate on SSP-3D with:
```
python run_evaluate.py -D ssp3d
```

Download 3DPW from [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/). You will need to preprocess the dataset first, to extract centred+cropped images and SMPL labels (adapted from [SPIN](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)):
```
python data/pw3d_preprocess.py --dataset_path $3DPW_DIR_PATH
```
This should create a subdirectory with preprocessed files, such that the 3DPW directory has the following structure:
```
$3DPW_DIR_PATH
      ├── test                                  
      │   ├── 3dpw_test.npz    
      │   ├── cropped_frames   
      ├── imageFiles
      └── sequenceFiles
```
Additionally, download HRNet 2D joint detections on 3DPW from [here](https://drive.google.com/drive/u/0/folders/1GnVukI3Z1h0fq9GeD40RI8z35EfKWEda), and place this in `$3DPW_DIR_PATH/test`. Update `configs/paths.py` with the path pointing to `$3DPW_DIR_PATH/test`. Evaluate on 3DPW with:
```
python run_evaluate.py -D 3dpw
```
The number of samples used to evaluate sample-related metrics can be changed using the `--num_samples` option (default is 10).

## TODO
- Training Code
- Gendered pre-trained models for improved shape estimation
- Weaknesses and future research

## Acknowledgments
Code was adapted from/influenced by the following repos - thanks to the authors!

- [HMR](https://github.com/akanazawa/hmr)
- [SPIN](https://github.com/nkolot/SPIN)
- [VIBE](https://github.com/mkocabas/VIBE)
- [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Probabilistic Orientation Estimation with Matrix Fisher Distributions](https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions)
- [CannyEdgePytorch](https://github.com/DCurro/CannyEdgePytorch)
- [Matrix-Fisher-Distribution](https://github.com/tylee-fdcl/Matrix-Fisher-Distribution)
