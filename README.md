# KDT Final Project

## Occasion

Cars are equipped with cameras. We want to build ADAS.
Use cameras on board to assist the human driver or to self-drive

## Goals

1. (Main) Detect where the free space is on the road _As Good As Tesla_
1. (Optional) Detect lane lines _As Good As Tesla_

## Obectives

1. Color the free space on the road by "**image segmentation techinque**"
1. Research state-of-the-art method to detect the free space on the road
1. Decide the proper image segmentation techinque for the project among many
1. Use dataset obtained from real car

## Image Segmentation Technique Candidates

1. W-net (Fully Unsupervised Image Segmentation)

   - [Xide Xia and Brian Kulis, W-Net: A Deep Model for Fully Unsupervised Image Segmentation (2017)](https://arxiv.org/abs/1711.08506)
   - <https://github.com/Andrew-booler/W-Net>

1. FCN-8 (Fully Convolutional Networks)

   - [Evan Shelhamer, et al., Fully Convolutional Networks for Semantic Segmentation (2016)](https://arxiv.org/abs/1605.06211)
   - [Jonathan Long, et al., Fully Convolutional Networks for Semantic Segmentation (2015)](https://arxiv.org/abs/1411.4038)
   - [Luis C. García-Peraza-Herrera, Real-Time Segmentation of Non-Rigid Surgical Tools based on Deep Learning and Tracking (2020)](https://arxiv.org/abs/2009.03016)
   - [Karen Simonyan and Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition (2015)](https://arxiv.org/abs/1409.1556)

1. SCNN (Spacial CNN)

   - [Angshuman Parashar, et al., SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks ()](https://arxiv.org/abs/1708.04485)
   - [Xingang Pan, et al., Spatial As Deep: Spatial CNN for Traffic Scene Understanding (2017)](https://arxiv.org/abs/1712.06080)
   - <https://github.com/XingangPan/SCNN>

## Dataset Candidates

- [42dot lane dataset](https://42dot.ai/akit/dataset/sdlane)
- [Kitti road dataset](https://www.cvlibs.net/datasets/kitti/eval_road.php)
