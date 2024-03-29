# Geometrical Scene Characterization for 6-DoF Camera Pose Estimation

## Overview
This Master's project focuses on identifying a camera's six degrees of freedom (6-DoF) pose in world coordinates using object detection techniques. It is crucial in robotics and computer vision applications, facilitating accurate object positioning within a global frame.

### Key Features
- **Camera Pose Estimation**: Determine the camera's position and orientation in a global frame.
- **3D Object Detection**: Utilize pre-trained models for detecting objects and estimating their orientation and spatial translation.
- **Extrinsic Parameter Identification**: Calculate camera extrinsic parameters based on object detection outcomes.

### Prerequisites
- Python 3.7
- Robot Operating System (ROS)
- Libraries: NumPy, OpenCV

## Architecture 

<img width="570" alt="image" src="https://github.com/rahulkstk/Thesis_calibration/assets/84446317/4b640777-04f2-4704-8a7a-1ad048095629">

Yolo darknet node: https://github.com/rahulkstk/darknet/blob/master/zed_sub_darknet.py
## Sample output


<img width="521" alt="image" src="https://github.com/rahulkstk/Thesis_calibration/assets/84446317/ceb33944-55ca-430f-b5eb-aac2bdab4c10">

## References

This project utilizes mediapipe objectron: https://github.com/google/mediapipe/blob/master/docs/solutions/objectron.md

