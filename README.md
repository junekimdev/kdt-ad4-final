# KDT Final Project

## Occasion

1. Cars are equipped with cameras.
1. We want to build ADAS.
1. Use cameras on board to assist the human driver or to self-drive

## User Stories

- As an ADS, it needs to plan ahead so that it can move safely in RMC.
  - As a planning engineer, I want to know where the free space is on the road.
- As an ADS, it needs to keep lane so that it can drive safely.
  - As a planning engineer, I want to know where the lane lines are.
- As an ADS, it needs to change lane so that it can go to destination.
  - As a planning engineer, I want to know where the lane lines are.

> ### Terms
>
> - ADS: Automated Driving System
> - RMC: Minimum Risk Condition

## Goals

1. (Main) Detect where the free space is on the road _As Good As Tesla_
   1. Color the free space of the road on the image by "**image segmentation techinque**"
1. (Optional) Detect lane lines _As Good As Tesla_
   1. Draw continous lines/curves which indicates lane lines

## Tasks

1. Research state-of-the-art method to detect the free space on the road
1. Decide the proper image segmentation techinque for the project among many
1. Use dataset obtained from real car
1. Build ML model and train/eval/test
   - Implement Custom Pytorch Dataset
   - Implement Custom Pytorch Dataloader
   - Implement Custom Pytorch module
   - Get images from dataset
   - Annotate date
   - Augment dataset
   - Implement unit tests
1. Implement MLOps pipeline
   - Carry out unit tests
   - Turn Pytorch model to ONNX model
   - Turn ONNX model to TensorRT model
   - Implement integration test
   - Carry out integration test
   - Deploy to Jetson TX2

## Candidates for Image Segmentation Technique

1. W-net (Fully Unsupervised Image Segmentation)

   - [Xide Xia and Brian Kulis, W-Net: A Deep Model for Fully Unsupervised Image Segmentation (2017)](https://arxiv.org/abs/1711.08506)
   - Codes: <https://github.com/Andrew-booler/W-Net>

1. FCN-8 (Fully Convolutional Networks)

   - [Evan Shelhamer, et al., Fully Convolutional Networks for Semantic Segmentation (2016)](https://arxiv.org/abs/1605.06211)
   - [Jonathan Long, et al., Fully Convolutional Networks for Semantic Segmentation (2015)](https://arxiv.org/abs/1411.4038)
   - [Luis C. García-Peraza-Herrera, Real-Time Segmentation of Non-Rigid Surgical Tools based on Deep Learning and Tracking (2020)](https://arxiv.org/abs/2009.03016)
   - [Karen Simonyan and Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition (2015)](https://arxiv.org/abs/1409.1556)

1. SCNN (Spacial CNN)

   - [Xingang Pan, et al., Spatial As Deep: Spatial CNN for Traffic Scene Understanding (2017)](https://arxiv.org/abs/1712.06080)
   - Codes: <https://github.com/XingangPan/SCNN>

## Dataset Candidates

- [42dot lane dataset](https://42dot.ai/akit/dataset/sdlane)
- [Kitti road dataset](https://www.cvlibs.net/datasets/kitti/eval_road.php)
