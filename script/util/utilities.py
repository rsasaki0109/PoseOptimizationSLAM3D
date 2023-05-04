import numpy as np
from .pose import Pose3D, Constraint3D

def plot_nodes(nodes, ax, color="-r", label=""):
    x, y, z = [], [], []
    for n in nodes:
        x.append(n.x)
        y.append(n.y)
        z.append(n.z)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.plot(x, y, z, color, label=label)


def load_data(fname):
    nodes, consts = [], []

    for line in open(fname):
        sline = line.split()
        tag = sline[0]

        if tag == "VERTEX_SE3:QUAT":
            # data_id = int(sline[1]) # unused
            x = float(sline[2])
            y = float(sline[3])
            z = float(sline[4])
            qx = float(sline[5])
            qy = float(sline[6])
            qz = float(sline[7])
            qw = float(sline[8])

            nodes.append(Pose3D(x, y, z, qw, qx, qy, qz))
        elif tag == "EDGE_SE3:QUAT":
            id1 = int(sline[1])
            id2 = int(sline[2])
            x = float(sline[3])
            y = float(sline[4])
            z = float(sline[5])
            qx = float(sline[6])
            qy = float(sline[7])
            qz = float(sline[8])
            qw = float(sline[9])
            c1 = float(sline[10])
            c2 = float(sline[11])
            c3 = float(sline[12])
            c4 = float(sline[13])
            c5 = float(sline[14])
            c6 = float(sline[15])
            c7 = float(sline[16])
            c8 = float(sline[17])
            c9 = float(sline[18])
            c10 = float(sline[19])
            c11 = float(sline[20])
            c12 = float(sline[21])
            c13 = float(sline[22])
            c14 = float(sline[23])
            c15 = float(sline[24])
            c16 = float(sline[25])
            c17 = float(sline[26])
            c18 = float(sline[27])
            c19 = float(sline[28])
            c20 = float(sline[29])
            c21 = float(sline[30])
            t = Pose3D(x, y, z, qw, qx, qy, qz)
            info_mat = np.array([[c1, c2, c3, c4, c5, c6],
                                 [c2, c7, c8, c9, c10, c11],
                                 [c3, c8, c12, c13, c14, c15],
                                 [c4, c9, c13, c16, c17, c18],
                                 [c5, c10, c14, c17, c19, c20],
                                 [c6, c11, c15, c18, c20, c21]
                                 ])
            consts.append(Constraint3D(id1, id2, t, info_mat))

    print("n_nodes:", len(nodes))
    print("n_consts:", len(consts))

    return nodes, consts