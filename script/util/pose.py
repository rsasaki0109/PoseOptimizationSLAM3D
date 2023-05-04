import numpy as np

def skew_symmetric(v):
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

class Quaternion:

    def __init__(self, qw, qx, qy, qz):
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def conjugate(self):
        return Quaternion(self.qw, -self.qx, -self.qy, -self.qz)

    def to_np(self):
        return np.array([self.qw, self.qx, self.qy, self.qz]).reshape(4, 1)

    def q_mult(self, q, out='np'):
        v = np.array([self.qx, self.qy, self.qz]).reshape(3, 1)
        sum_term = np.zeros([4, 4])
        sum_term[0, 1:] = -v[:, 0]
        sum_term[1:, 0] = v[:, 0]
        sum_term[1:, 1:] = skew_symmetric(v)
        sigma = self.qw * np.eye(4) + sum_term
        q_new = sigma @ q.to_np()
        if out == 'np':
            return q_new
        elif out == 'Quaternion':
            q_obj = Quaternion(*q_new)
            return q_obj

class RotVec:

    def __init__(self, ax=0., ay=0., az=0., quaternion=None):
        if quaternion is None:
            self.ax = ax
            self.ay = ay
            self.az = az
        else:
            x = quaternion.qx
            y = quaternion.qy
            z = quaternion.qz
            norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            if norm < 1e-7:
                self.ax = 2 * x
                self.ay = 2 * y
                self.az = 2 * z
            else:
                th = 2 * np.arctan2(norm, quaternion.qw)
                th = self.pi2pi(th)
                self.ax = x / norm * th
                self.ay = y / norm * th
                self.az = z / norm * th

    def inverted(self):

        return RotVec(-self.ax, -self.ay, -self.az)

    def to_rotation_matrix(self):

        q = self.to_quaternion()
        q_vec = np.array([q.qx, q.qy, q.qz]).reshape(3, 1)
        qw = q.qw
        mat = (qw ** 2 - q_vec.T @ q_vec) * np.eye(3) \
              + 2 * q_vec @ q_vec.T - 2 * qw * skew_symmetric(q_vec.reshape(-1, ))
        return mat.T

    def to_quaternion(self):

        v = np.sqrt(self.ax ** 2 + self.ay ** 2 + self.az ** 2)
        if (v < 1e-6):
            return Quaternion(1, 0, 0, 0)
        else:
            return Quaternion(np.cos(v / 2), np.sin(v / 2) * self.ax / v,
                              np.sin(v / 2) * self.ay / v, np.sin(v / 2) * self.az / v)

    @staticmethod
    def pi2pi(rad):

        val = np.fmod(rad, 2.0 * np.pi)
        if val > np.pi:
            val -= 2.0 * np.pi
        elif val < -np.pi:
            val += 2.0 * np.pi

        return val

class TripletList:

    def __init__(self):
        self.row = []
        self.col = []
        self.data = []

    def push_back(self, irow, icol, idata):
        self.row.append(irow)
        self.col.append(icol)
        self.data.append(idata)

class Pose3D:

    def __init__(self, x, y, z, qw, qx, qy, qz):
        self.x = x
        self.y = y
        self.z = z
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def pos(self):
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        return v

    def rv(self):
        q = Quaternion(self.qw, self.qx, self.qy, self.qz)
        return RotVec(quaternion=q)

    def ominus(self, base):
        t = base.rv().to_rotation_matrix().T @ (self.pos() - base.pos())
        q = base.rv().to_quaternion().conjugate().q_mult(self.rv().to_quaternion(), out='Quaternion')
        return Pose3D(t[0][0], t[1][0], t[2][0], q.qw, q.qx, q.qy, q.qz)

class Constraint3D:

    def __init__(self, id1, id2, t, info_mat):
        self.id1 = id1
        self.id2 = id2
        self.t = t
        self.info_mat = info_mat
