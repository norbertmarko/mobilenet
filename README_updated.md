# MobileNet v1 + SkipNet

The purpose of the project is to create a model capable of real-time semantic segmentation for autonomous driving tasks. This particular model focuses mainly on the drivable surface of the drive scene, trained on two different datasets ([Cityscapes](https://arxiv.org/abs/1604.01685), [BDD100k](https://arxiv.org/abs/1805.04687)).


metrics measuring, demo videos ( segmentation, lidar), .sh start - things to run it, not the training stuff)

PUT DEMO GIF HERE 3D POINT CLOUD AND LANE SEGMENTATION


https://drive.google.com/drive/folders/1VrDMp4ggWMrSRs_SPlASP4Z8Hwmq_ubn?usp=sharing

## Table of Contents

+ [The Goal](#the-goal)
+ [Tools](#tools)
+ [Planned Task List](#planned-task-list)
+ [Installation](#installation)
+ [Usage](#usage)
+ [ROS integration](#ros-integration)
+ [Results](#results)
+ [Pre-trained Weights](#pre-trained-weights)
+ [Acknowledgements](#acknowledgements)


## The Goal

Most of the research on semantic segmentation focuses on increasing the accuracy of the models while computationally efficient solutions are hard to come by and even then, huge compromises have to be made.

Since the goal is clear that we are looking at basic autonomous driving, the model focuses on adequate accuracy on the road, and its boundaries (this can be expanded) so we can estimate our drivable surface or the individual lanes.

To achieve this goal, I implemented a custom architecture with separate feature extraction and decoder modules. The inspiration for this work comes from [this paper](#realtime).

Optimization is done with TensorRT to achieve the fastest inference speed possible for the model so it can be used for testing in autonomous vehicles. Since the repository uses TensorFlow

## Installation
```
pip3 install -r requirements.txt
```


## Usage

Here goes the commands to start with / without tensorrt -> individual image, video stream, capture video, ROS etc. [.sh file to start it on our car!!! (+ instruction for these to starting from the repo + weights DL, with the metrics shown)!!!]

## Evaluate Metrics

## 3D Point Cloud
put this into results to

## Results

video, images

## Pre-trained Weights

The model has been trained on the Berkeley Deep Drive and the Cityscapes dataset. Using the former weights results in lane segmentation, while the latter achieves drivable surface segmentation without differentiating the lanes. The weights trained on different input sizes can be found in the table below.

| Dataset      | Size (H x W) |
| :---        |    :----:   |          ---: |
| Cityscapes      | 288 x 512 ([Google Drive](https://drive.google.com/drive/folders/1VrDMp4ggWMrSRs_SPlASP4Z8Hwmq_ubn?usp=sharing)), 376 x 672 (Google Drive)        | 
| BDD100k   | 288 x 512 ([Google Drive](https://drive.google.com/drive/folders/1VrDMp4ggWMrSRs_SPlASP4Z8Hwmq_ubn?usp=sharing)), 376 x 672 ([Google Drive](https://drive.google.com/drive/folders/1VrDMp4ggWMrSRs_SPlASP4Z8Hwmq_ubn?usp=sharing))       |


## Acknowledgements

<a id="realtime">["RTSeg: Real-time Semantic Segmentation Comparative Study"](https://arxiv.org/abs/1803.02758) by Mennatullah Siam, Mostafa Gamal, Moemen Abdel-Razek, Senthil Yogamani, Martin Jagersand</a>

<a id="mobile1">["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam</a>

<a id="inception">["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna</a>

<a id="fcn8">["Fully Convolutional Networks for Semantic Segmentation"](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell</a>

<a id="googlebrain">["Revisiting Distributed Synchronous SGD"](https://arxiv.org/abs/1604.00981) by Jianmin Chen, Xinghao Pan, Rajat Monga, Samy Bengio, Rafal Jozefowicz</a>