# Visual SLAM

## Introduction

Most of the implementations are inspired by the framework provided by the course [Vision-based Navigation IN2106](https://vision.in.tum.de/teaching/ws2022/visnav_ws2022) from my university (TUM). In the course, we only finished visual odometry, and I would like to add a loop closure module and relocalization module to make it become a more sophisticated SLAM sytem. For more detial about how I implement these modules in detail, please refer to my project page here [Visual-SLAM: Loop Closure and Relocalization](https://hip-fin-125.notion.site/Visual-SLAM-Loop-Closure-and-Relocalization-ef7be594875a47e598cf261b64e9b684).

## Setup

```
git clone --recursive https://github.com/yunjinli/visual-slam.git
```

```
cd visual-slam
./install_dependencies.sh
./build_submodules.sh
```
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make DBoW2
make slam
```

## Run
Note: It's only tested to load EuRoC Machine Hall and EuRoc Vicon Room1 datasets
```
cd build
./slam --dataset-path ../data/<dataset_name>/mav0/ --cam-calib ../calibration_file/<calibration_file_name>
```

## Quantitative Evaluation
### My SLAM
||MH01|MH02|MH03|MH04|MH05|V1_1|V1_2|V1_3|
|----|---|---|---|---|---|---|---|---|
|ATE (m)|**0.200**|0.364|**0.782**|**0.890**|**1.046**|**0.090**|**0.197**|**0.551**|
### Baseline VO
||MH01|MH02|MH03|MH04|MH05|V1_1|V1_2|V1_3|
|----|---|---|---|---|---|---|---|---|
|ATE (m)|1.152|**0.305**|3.734|4.330|12.930|0.113|4.355|6.184|

## Visualization Result
### EuRoC Vicon Room 1 (V1_1_easy)
![V1_1_3](/picture/V1_2_3.png)
### EuRoC Vicon Room 1 (V1_2_medium)
![...](/picture/...)
### EuRoC Vicon Room 1 (V1_3_difficult)
![...](/picture/...)
### EuRoC Machine Hall 1 (MH_01_easy)
![...](/picture/...)
### EuRoC Machine Hall 2 (MH_02_easy)
![...](/picture/...)
### EuRoC Machine Hall 3 (MH_03_medium)
![...](/picture/...)
### EuRoC Machine Hall 4 (MH_04_difficult)
![...](/picture/...)
### EuRoC Machine Hall 5 (MH_05_difficult)
![...](/picture/...)

## Video Demo
### Machine Hall 04
[![MH04](https://img.youtube.com/vi/aNgcuXywrX4/0.jpg)](https://youtu.be/aNgcuXywrX4)

## References

Mur-Artal, Raul, Jose Maria Martinez Montiel, and Juan D. Tardos. "ORB-SLAM: a versatile and accurate monocular SLAM system." IEEE transactions on robotics 31.5 (2015): 1147-1163.

R. Mur-Artal and J. D. Tardós, "ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras," in IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255-1262, Oct. 2017, doi: 10.1109/TRO.2017.2705103.

R. Mur-Artal and J. D. Tardós, "Fast relocalisation and loop closing in keyframe-based SLAM," 2014 IEEE International Conference on Robotics and Automation (ICRA), Hong Kong, China, 2014, pp. 846-853, doi: 10.1109/ICRA.2014.6906953.

Strasdat, Hauke, J. Montiel, and Andrew J. Davison. "Scale drift-aware large scale monocular SLAM." Robotics: Science and Systems VI 2.3 (2010): 7.
