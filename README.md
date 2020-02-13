# MobileNet v1/v2 + SkipNet

The purpose of the project is to create a model capable of real-time semantic segmentation with sufficient accuracy to estimate the drivable surface and the obstacles in front of the vehicle. This repository will act as a continuous documentation for the project.

## Table of Contents

+ [The Goal](#the-goal)
+ [Tools](#tools)
+ [Planned Task List](#planned-task-list)
+ [ROS integration](#ros-integration)
+ [Acknowledgements](#acknowledgements)

## The Goal

Most of the research on semantic segmentation is focusing on increasing the accuracy of the models while computationally efficient solutions are hard to find and many times huge compromises have to be made.

Since the goal is clear that we are looking at basic autonomous driving, the model only needs adequate accuracy on the road, boundary and obstacle (this can be expanded) labels so we can estimate our drivable surface.

To achieve this goal, I will try to implement a custom architecture with separate feature extraction and decoder modules. The inspiration for this work comes from [this paper](#realtime).

After the model is done, optimization will be done with TensorRT to achieve the fastest inference speed possible for the model so it can be used for testing in autonomous vehicles.



![Custom Encoder-Decoder](https://github.com/norbertmarko/mobilenet/blob/master/docs/figures/encoder_decoder.png)


The feature extractor will be a modified Inception architecture  ([MobileNet v1](#mobile1) and [MobileNet v2](#mobile2)). The main difference is that [Inception V3](#inception) uses standard convolution while MobileNet employs depthwise separable convolution. The decoder will follow the standard FCN-8 architecture([SkipNet](#fcn8)) 

## Tools

The main tool used during the project is Google's TensorFlow, with consideration to the supported operations recommendation in the [TensorRT documentation](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#supported-ops).

The aforementioned TensorRT by NVIDIA will serve as the optimizer to achieve faster inference.

For training and fine-tuning the models, 3 GeForce RTX 2080 GPUs are used simultaneously. Recommendations for the multi GPU training are used from [this Google Brain study](#googlebrain) (synchronous data parallelism)


## Planned Task List

```

- [x] Creating the pipeline in TensorFlow
- [x] Use a pre-trained MobileNet v1 (ImageNet) with SkipNet decoder for fine tuning
- [ ] Use a pre-trained MobileNet v2 (ImageNet) with SkipNet decoder for fine tuning
- [ ] Optimize the models in TensorRT
- [ ] Compare performances and publish the results here
- [ ] Add single frame inference results
- [ ] Add video inference results

```

## ROS integration

Running the network wrapped into ROS (Robotic Operating System) on Ubuntu:

1. Copy the "mobilenet-ros" package from /test/integration into your workspace's 'src' folder. ([Creating a workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace))

```
$ cd ~/catkin_ws/src
```
2. Go back to your root workspace folder to build the package then source it.

```
$ cd ../
$ catkin_make
$ source ./devel/setup.bash
```
3. Download the trained model ([Model on Google Drive]()), and place it into the 'mobilenet-ros/src/model' directory. Specify the absolute path in 'mobilenet-ros/src/scripts/conf.py' by changing the
'trt_opt_model' variable to the new path. Use the commands below to find out the absolute path on your machine. Put the model name you placed in the model directory after 'realpath' command.

```
$ cd ~/catkin_ws/src/mobilenet-ros/src/model
$ realpath freezed_model_trt.pb
```
4. You will need several terminal windows for this stage (Terminator recommended). Start ROS core in one, open two more terminals, navigate to your workspace root and source in both of them the way you did in step 2.
Go to 'src/mobilenet-ros/src/scripts' in one of the workspace root windows and make the 'camera_node.py' and 'tf_ros_predict.py' scripts exacutable. Navigate back to workspace root when you are done.

```
# Terminal 1
$ roscore

# Terminal 2
$ cd ~/catkin_ws/
$ source ./devel/setup.bash

# Terminal 3
$ cd ~/catkin_ws/
$ source ./devel/setup.bash
$ cd ./src/mobilenet-ros/src/scripts
$ chmod +x camera_node.py
$ chmod +x tf_ros_predict.py
$ cd ../../../..
```
5. Use your two workspace root terminal windows to run the camera node in one and the segmentation subscriber in the other. Make sure you have a webcam plugged in. If you use a camera that is capable of publishing
it's image frames as a ROS topic, then change the topic name in the conf.py file and run the camera first.

```
$ rosrun mobilenet-ros camera_node.py
$ rosrun mobilenet-ros tf_ros_predict.py
```


## Acknowledgements

<a id="realtime">["RTSeg: Real-time Semantic Segmentation Comparative Study"](https://arxiv.org/abs/1803.02758) by Mennatullah Siam, Mostafa Gamal, Moemen Abdel-Razek, Senthil Yogamani, Martin Jagersand</a>

<a id="mobile1">["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam</a>

<a id="mobile2">["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen</a>

<a id="inception">["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna</a>

<a id="fcn8">["Fully Convolutional Networks for Semantic Segmentation"](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, Trevor Darrell</a>

<a id="googlebrain">["Revisiting Distributed Synchronous SGD"](https://arxiv.org/abs/1604.00981) by Jianmin Chen, Xinghao Pan, Rajat Monga, Samy Bengio, Rafal Jozefowicz</a>