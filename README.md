# Video recognition
This repository aims to replicate the main results from several seminal papers in the field of action recognition on videos as well as try out new architectures. All models can be found in the `keras_models.py` file.
## Reference papers
> 
**Two-Stream Convolutional Networks for Action Recognition in Videos**,
Karen Simonyan, Andrew Zisserman,
*NIPS 2014*.
[[Arxiv Preprint](https://arxiv.org/pdf/1406.2199.pdf)]
> 
**Temporal Segment Networks: Towards Good Practices for Deep Action Recognition**,
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool,
*ECCV 2016*.
[[Arxiv Preprint](http://arxiv.org/abs/1608.00859)]
> 
**Two-Stream RNN/CNN for Action Recognition in 3D Videos**,
Rui Zhao, Haider Ali, Patrick van der Smagt,
*2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.
[[Arxiv Preprint](https://arxiv.org/pdf/1703.09783.pdf)]
> 
**VideoLSTM Convolves, Attends and Flows for Action Recognition**,
Zhenyang Li, Efstratios Gavves, Mihir Jain, Cees G. M. Snoek,
*Computer Vision and Image Understanding 2018*.
[[Arxiv Preprint](https://arxiv.org/pdf/1607.01794.pdf)]

## 1. Data
  ### 1.1 UCF101 Dataset 
  Download the preprocessed data directly from [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion))
  * RGB images
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003
  
  cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
  unzip ucf101_jpegs_256.zip
  ```
  * Optical Flow
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  
  cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  unzip ucf101_tvl1_flow.zip
  ```
  ### 1.2 PennAction Dataset
  Available for download via the following link: https://upenn.box.com/PennAction
  ### 1.3 MyVideos Dataset
  Consists of 80 own recorded videos on 5 different actions (PickUpObject, TakeOffShoes, TakeOffClothes, Stealing and NotStealing). RGB   frames and human poses from each video in this dataset were extracted using `OpenCV` and `CMU's OpenPose` libraries.
