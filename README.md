# PoseOptimizationSLAM3D
3D (x, y, z, qw, qx, qy, qz) pose optimization SLAM

1. Download data  

~~~
python script/data_downloader.py
~~~

2. run SLAM 

~~~
python script/pose_optimization_slam_3d.py
~~~

# Result

parking-garage.g2o data

step 1  
![demo](./images/parking-garage_step1.png)  

step 10  
![demo](./images/parking-garage_step10.png)  
 
standard output  
![demo](./images/output.png)  

sphere2200_guess.g2o  
![demo](./images/sphere2200_guess.g2o.png)  

torus3d_guess.g2o  
![demo](./images/torus3d_guess.g2o.png)  

# Reference 
[A Compact and Portable Implementation of Graph\-based SLAM](https://www.researchgate.net/publication/321287640_A_Compact_and_Portable_Implementation_of_Graph-based_SLAM)  
[GitHub \- furo\-org/p2o: Single header 2D/3D graph\-based SLAM library](https://github.com/furo-org/p2o)  
[GitHub \- AtsushiSakai/PythonRobotics
/SLAM/PoseOptimizationSLAM](https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/PoseOptimizationSLAM)
