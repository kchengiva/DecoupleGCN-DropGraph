# DecoupleGCN-DropGraph
The implementation for "Decoupling GCN with DropGraph Module for Skeleton-Based Action Recognition" (ECCV2020). The proposed method boosts the performance of spatial-temporal graph convolutional network with NO extra FLOPs, NO extra latency, and NO extra GPU memory cost.

## Prerequisite

 - PyTorch 0.4.1
 - Cuda 9.0

## Data Preparation

 - Download the raw data of [NTU-RGBD](https://github.com/shahroudy/NTURGB-D) and [NTU-RGBD120](https://github.com/shahroudy/NTURGB-D). Put NTU-RGBD data under the directory `./data/nturgbd_raw`. Put NTU-RGBD120 data under the directory `./data/nturgbd120_raw`. 
 
 - For NTU-RGBD, preprocess data with `python data_gen/ntu_gendata.py`. For NTU-RGBD120, preprocess data with `python data_gen/ntu120_gendata.py`. 
  
 - Generate the bone data with `python data_gen/gen_bone_data.py`.

 - Generate the motion data with `python data_gen/gen_motion_data.py`.

## Training & Testing

  - NTU X-view

    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_joint_motion.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone_motion.yaml`

  - NTU X-sub

    `python main.py --config ./config/nturgbd-cross-subject/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_joint_motion.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_bone_motion.yaml`

  - For NTU120, change the dataset path in config files, and change `num_class` in config files from 60 to 120.
  
## Multi-stream ensemble

To ensemble the results of 4 streams. Change models name in `ensemble.py` depending on your experiment setting. Then run `python ensemble.py`.

## Trained models

We release several trained models:

Model|Dataset|Setting|Top1(%)
-|-|-|-
./save_models/ntu_joint_xview.pt|NTU-RGBD|X-view|95.2
./save_models/ntu_joint_xsub.pt|NTU-RGBD|X-sub|88.2
./save_models/ntu120_joint_xsetup.pt|NTU-RGBD120|X-setup|84.3
./save_models/ntu120_joint_xsub.pt|NTU-RGBD120|X-sub|82.4

     
## Citation
If you find this model useful for your resesarch, please use the following BibTeX entry.

    @inproceedings{cheng2020eccv,  
      title     = {Decoupling GCN with DropGraph Module for Skeleton-Based Action Recognition},  
      author    = {Ke Cheng and Yifan Zhang and Congqi Cao and Lei Shi and Jian Cheng and Hanqing Lu},  
      booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},  
      year      = {2020},  
    }
