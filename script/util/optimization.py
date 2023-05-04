import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from .pose import Pose3D, TripletList, RotVec, Quaternion

class Optimizer3D:

    def __init__(self):
        self.verbose = False
        self.animation = False
        self.p_lambda = 1e-7
        self.init_w = 1e10
        self.stop_thre = 1e-3
        self.robust_delta = 1
        self.dim = 6  # state dimension

    def optimize_path(self, nodes, consts, max_iter, min_iter):

        graph_nodes = nodes[:]
        prev_cost = sys.float_info.max

        est_traj_fig = plt.figure()
        ax = est_traj_fig.add_subplot(111, projection='3d')

        for i in range(max_iter):
            start = time.time()
            cost, graph_nodes = self.optimize_path_one_step(
                graph_nodes, consts)
            elapsed = time.time() - start
            if self.verbose:
                print("step ", i, " cost: ", cost, " time:", elapsed, "s")

            # check convergence
            if (i > min_iter) and (prev_cost - cost < self.stop_thre):
                if self.verbose:
                    print("converged:", prev_cost
                          - cost, " < ", self.stop_thre)
                    break
            prev_cost = cost

            if self.animation:
                plt.cla()
                plot_nodes(nodes, ax, color="-b")
                plot_nodes(graph_nodes, ax)
                plt.pause(1.0)

        return graph_nodes

    def optimize_path_one_step(self, graph_nodes, constraints):

        indlist = [i for i in range(self.dim)]
        numnodes = len(graph_nodes)
        bf = np.zeros(numnodes * self.dim)
        tripletList = TripletList()

        for con in constraints:
            ida = con.id1
            idb = con.id2
            assert 0 <= ida < numnodes, "ida is invalid"
            assert 0 <= idb < numnodes, "idb is invalid"
            pa = graph_nodes[ida]
            pb = graph_nodes[idb]
            r, Ja, Jb = self.calc_error(
                pa, pb, con.t)

            info_mat = con.info_mat * self.robust_coeff(r.reshape(self.dim, 1).T @ con.info_mat @ r.reshape(self.dim, 1),
                                                   self.robust_delta)

            trJaInfo = Ja.transpose() @ info_mat
            trJaInfoJa = trJaInfo @ Ja
            trJbInfo = Jb.transpose() @ info_mat
            trJbInfoJb = trJbInfo @ Jb
            trJaInfoJb = trJaInfo @ Jb

            for k in indlist:
                for m in indlist:
                    tripletList.push_back(
                        ida * self.dim + k, ida * self.dim + m, trJaInfoJa[k, m])
                    tripletList.push_back(
                        idb * self.dim + k, idb * self.dim + m, trJbInfoJb[k, m])
                    tripletList.push_back(
                        ida * self.dim + k, idb * self.dim + m, trJaInfoJb[k, m])
                    tripletList.push_back(
                        idb * self.dim + k, ida * self.dim + m, trJaInfoJb[m, k])

            bf[ida * self.dim: (ida + 1) * self.dim] += trJaInfo @ r
            bf[idb * self.dim: (idb + 1) * self.dim] += trJbInfo @ r

        for k in indlist:
            tripletList.push_back(k, k, self.init_w)

        for i in range(self.dim * numnodes):
            tripletList.push_back(i, i, self.p_lambda)

        mat = sparse.coo_matrix((tripletList.data, (tripletList.row, tripletList.col)),
                                shape=(numnodes * self.dim, numnodes * self.dim))

        x = linalg.spsolve(mat.tocsr(), -bf)

        out_nodes = []

        for i in range(len(graph_nodes)):
            u_i = i * self.dim

            q_before = Quaternion(graph_nodes[i].qw, graph_nodes[i].qx, graph_nodes[i].qy, graph_nodes[i].qz)
            rv_before = RotVec(quaternion=q_before)
            rv_after = RotVec(ax=rv_before.ax + x[u_i + 3], ay=rv_before.ay + x[u_i + 4], az=rv_before.az + x[u_i + 5])
            q_after = rv_after.to_quaternion()

            pos = Pose3D(
                graph_nodes[i].x + x[u_i],
                graph_nodes[i].y + x[u_i + 1],
                graph_nodes[i].z + x[u_i + 2],
                q_after.qw,
                q_after.qx,
                q_after.qy,
                q_after.qz
            )
            out_nodes.append(pos)

        cost = self.calc_global_cost(out_nodes, constraints)

        return cost, out_nodes

    def calc_global_cost(self, nodes, constraints):

        cost = 0.0
        for c in constraints:
            diff = self.error_func(nodes[c.id1], nodes[c.id2], c.t)
            info_mat = c.info_mat * self.robust_coeff(diff.reshape(self.dim, 1).T @ c.info_mat @ diff.reshape(self.dim, 1),
                                                 self.robust_delta)
            cost += diff.transpose() @ info_mat @ diff

        return cost

    @staticmethod
    def error_func(pa, pb, t):

        ba = pb.ominus(pa)
        q = t.rv().to_quaternion().conjugate().q_mult(ba.rv().to_quaternion(), out='Quaternion')
        drv = RotVec(quaternion=q)
        error = np.array([ba.x - t.x,
                          ba.y - t.y,
                          ba.z - t.z,
                          drv.ax[0],
                          drv.ay[0],
                          drv.az[0]])
        return error

    @staticmethod
    def dQuat_dRV(rv):
        ax = rv.ax
        ay = rv.ay
        az = rv.az
        v = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        if v < 1e-6:
            dqu = 0.25 * np.array(
                [[-ax, -ay, -az],
                 [2.0, 0.0, 0.0],
                 [0.0, 2.0, 0.0],
                 [0.0, 0.0, 2.0]]
            )
            return dqu

        v2 = v ** 2
        v3 = v ** 3

        S = np.sin(v / 2.0)
        C = np.cos(v / 2.0)
        dqu = np.array(
            [[-ax * S / (2 * v), -ay * S / (2 * v), -az * S / (2 * v)],
             [S / v + ax * ax * C / (2 * v2) - ax * ax * S / v3, ax * ay * (C / (2 * v2) - S / v3),
              ax * az * (C / (2 * v2) - S / v3)],
             [ax * ay * (C / (2 * v2) - S / v3), S / v + ay * ay * C / (2 * v2) - ay * ay * S / v3,
              ay * az * (C / (2 * v2) - S / v3)],
             [ax * az * (C / (2 * v2) - S / v3), ay * az * (C / (2 * v2) - S / v3),
              S / v + az * az * C / (2 * v2) - az * az * S / v3]]
        )

        return dqu

    def dR_dRV(self, rv):

        q = rv.to_quaternion()
        qw = q.qw
        qx = q.qx
        qy = q.qy
        qz = q.qz
        dRdqw = 2 * np.array(
            [[qw, -qz, qy],
             [qz, qw, -qx],
             [-qy, qx, qw]]
        )
        dRdqx = 2 * np.array(
            [[qx, qy, qz],
             [qy, -qx, -qw],
             [qz, qw, -qx]]
        )
        dRdqy = 2 * np.array(
            [[-qy, qx, qw],
             [qx, qy, qz],
             [-qw, qz, -qy]]
        )
        dRdqz = 2 * np.array(
            [[-qz, -qw, qx],
             [qw, -qz, -qy],
             [qx, qy, qz]]
        )
        dqdu = self.dQuat_dRV(rv)
        dux = dRdqw * dqdu[0, 0] + dRdqx * dqdu[1, 0] + dRdqy * dqdu[2, 0] + dRdqz * dqdu[3, 0]
        duy = dRdqw * dqdu[0, 1] + dRdqx * dqdu[1, 1] + dRdqy * dqdu[2, 1] + dRdqz * dqdu[3, 1]
        duz = dRdqw * dqdu[0, 2] + dRdqx * dqdu[1, 2] + dRdqy * dqdu[2, 2] + dRdqz * dqdu[3, 2]
        return dux, duy, duz

    @staticmethod
    def dRV_dQuat(q):

        qw = q.qw[0]
        qx = q.qx[0]
        qy = q.qy[0]
        qz = q.qz[0]

        if 1 - qw ** 2 < 1e-7:
            ret = np.array(
                [[0.0, 2.0, 0.0, 0.0],
                 [0.0, 0.0, 2.0, 0.0],
                 [0.0, 0.0, 0.0, 2.0]]
            )
            return ret

        c = 1 / (1 - qw ** 2)
        d = np.arccos(qw) / (np.sqrt(1 - qw ** 2))
        ret = 2.0 * np.array(
            [[c * qx * (d * qw - 1), d, 0.0, 0.0],
             [c * qy * (d * qw - 1), 0.0, d, 0.0],
             [c * qz * (d * qw - 1), 0.0, 0.0, d]]
        )
        return ret

    def QMat(self, q):

        qw = q.qw
        qx = q.qx
        qy = q.qy
        qz = q.qz
        Q = np.array(
            [[qw, -qx, -qy, -qz],
             [qx, qw, -qz, qy],
             [qy, qz, qw, -qx],
             [qz, -qy, qx, qw]]
        )
        return Q

    def QMatBar(self, q):

        qw = q.qw
        qx = q.qx
        qy = q.qy
        qz = q.qz
        Q = np.array(
            [[qw, -qx, -qy, -qz],
             [qx, qw, qz, -qy],
             [qy, -qz, qw, qx],
             [qz, qy, -qx, qw]]
        )
        return Q

    def calc_error(self, pa, pb, t):

        e0 = self.error_func(pa, pb, t)
        Ja = np.identity(6)
        Jb = np.identity(6)

        rva_inv = pa.rv().inverted()
        rotPaInv = rva_inv.to_rotation_matrix()

        Ja[:3, :3] = -rotPaInv
        Jb[:3, :3] = rotPaInv

        dRux, dRuy, dRuz = self.dR_dRV(rva_inv)

        posdiff = np.array([[pb.x - pa.x], [pb.y - pa.y], [pb.z - pa.z]])

        Ja[0:3, 3:4] = -dRux @ posdiff
        Ja[0:3, 4:5] = -dRuy @ posdiff
        Ja[0:3, 5:6] = -dRuz @ posdiff

        # rotation part: qdiff = qc-1 * qa-1 * qb
        qainv = rva_inv.to_quaternion()
        qcinv = t.rv().inverted().to_quaternion()
        qb = pb.rv().to_quaternion()
        qinvca = qcinv.q_mult(qainv, out='Quaternion')
        qdiff = qinvca.q_mult(qb, out='Quaternion')

        Ja[3:6, 3:6] = -self.dRV_dQuat(qdiff) @ self.QMat(qcinv) @ self.QMatBar(qb) @ self.dQuat_dRV(rva_inv)
        Jb[3:6, 3:6] = self.dRV_dQuat(qdiff) @ self.QMat(qcinv) @ self.QMat(qainv) @ self.dQuat_dRV(pb.rv())

        return e0, Ja, Jb

    def robust_coeff(self, squared_error, delta):

        if squared_error < 0:
          return 0
        sqre = np.sqrt(squared_error)
        if sqre < delta:
          return 1  # no effect
        return delta / sqre  # linear
