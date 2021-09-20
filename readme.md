# 3D Pose Estimator

2D pose estimation part of this program is based on RMPE(https://github.com/Fang-Haoshu/RMPE)
3D pose estimation is based on [paper](https://arxiv.org/abs/1609.09058 "arXiv:1609.09058") by Ruiqi Zhao, Yan Wang and Aleix Martinez.

The program performs multiperson pose estimation in both 2D and 3D. The output is  stored as numpy array for both 2D and 3D, the 3D visualization is stored in gif format.
The 2D visualization is stored in .avi format.

Listed below are the steps to run the demo notebook.

## Clone the repository

## Install the requirements

## Download caffe model

Download the models from [here](https://drive.google.com/file/d/1muCh_8a4wdzuFAqTHnS2qRwp1AOhXP2K/view?usp=sharing "models")
Place the contents of detector folder to 'detector_model' folder in the cloned directory
Place the contents of estimator folder to '2d_model' folder in the cloned directory

## Run the 'demo.inpynb' notebook

Output of 2D skeleton tracking can be found in folder 'output2d'

![Screenshot](https://github.com/kumarvipu1/3D-Human-Pose-Estimation/blob/main/output2d/Capture.PNG)

Output of 3D tracking can be found in folder 'output3d'

![Screenshot3D](https://github.com/kumarvipu1/3D-Human-Pose-Estimation/blob/main/output3d/figures/56.png)
