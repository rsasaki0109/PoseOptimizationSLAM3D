"""
  This code is written by reference to [p2o(Petite Portable Pose-graph Optimizer)](https://github.com/furo-org/p2o)
  Copyright (C) 2010-2017 Kiyoshi Irie
  Copyright (C) 2017 Future Robotics Technology Center (fuRo),
                     Chiba Institute of Technology.

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this file,
  You can obtain one at https://mozilla.org/MPL/2.0/.


3D (x, y, z, qw, qx, qy, qz) pose optimization SLAM

author: Ryohei Sasaki(@rsasaki0109)

Ref:

- [A Compact and Portable Implementation of Graph\-based SLAM](https://www.researchgate.net/publication/321287640_A_Compact_and_Portable_Implementation_of_Graph-based_SLAM)

- [GitHub \- furo\-org/p2o: Single header 2D/3D graph\-based SLAM library](https://github.com/furo-org/p2o)

- [GitHub \- AtsushiSakai/PythonRobotics
/SLAM/PoseOptimizationSLAM](https://github.com/AtsushiSakai/PythonRobotics/blob/master/SLAM/PoseOptimizationSLAM/pose_optimization_slam.py)
"""

import time
import matplotlib.pyplot as plt
from util.utilities import plot_nodes, load_data
from util.optimization import Optimizer3D

def main():
    print("start!!")

    fnames = [
        "data/parking-garage.g2o",
        "data/sphere2200.g2o",
        "data/torus3d.g2o",
    ]

    max_iter = 10
    min_iter = 3

    optimizer = Optimizer3D()
    optimizer.p_lambda = 1e-6
    optimizer.verbose = True
    optimizer.animation = False

    for f in fnames:
        print(f)

        nodes, consts = load_data(f)

        start = time.time()
        final_nodes = optimizer.optimize_path(nodes, consts, max_iter, min_iter)
        print("elapsed_time", time.time() - start, "sec")

        plt.close("all")
        est_traj_fig = plt.figure()
        ax = est_traj_fig.add_subplot(111, projection='3d')
        plot_nodes(nodes, ax, color="-b", label="before")
        plot_nodes(final_nodes, ax, label="after")
        plt.legend()
        plt.show()

    print("done!!")

if __name__ == '__main__':
    main()