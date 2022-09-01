from data_loading import _DATA
import numpy as np
from utils import(
    rotation_matrix,
    or_lstsq,
    xy_to_xyf,
    timethis,
    profile,
    cal_q,
    drop,
)

LIM_ERR = _DATA().LIM_ERR_RO


class Relative_Orientation:
    def __init__(self, img_left, img_right, f, DIM) -> None:
        self.img_left = img_left
        self.img_right = img_right
        self.f = f
        # omiga1 equals zero constantly
        self.angle = np.expand_dims(np.array([0., 0., 0., 0., 0., 0.]), axis=0)
        self.DIM = DIM

    def img_space_coord_to_assisted_img_space_coord(self, xyf, R):
        return np.dot(R, xyf)

    def XYZf_to_AL(self, XYZ1, XYZ2, f):
        X1, X2 = XYZ1[0, :], XYZ2[0, :]
        Y1, Y2 = XYZ1[1, :], XYZ2[1, :]
        Z1, Z2 = XYZ1[2, :], XYZ2[2, :]
        a1 = np.expand_dims(-(X1*Y2)/Z1, axis=0)
        a2 = np.expand_dims(X1, axis=0)
        a3 = np.expand_dims((X2*Y1)/Z1, axis=0)
        a4 = np.expand_dims((Y1*Y2)/Z1-Z1, axis=0)
        # a4 = np.expand_dims((Y1*Y2)/Z1+Z1,axis=0)
        a5 = np.expand_dims(-X2, axis=0)
        A = np.concatenate((a1, a2, a3, a4, a5), axis=0).T
        # unsolved: why adding minus here
        L = np.expand_dims(-f*(Y1/Z1-Y2/Z2), axis=0).T
        return A, L

    @profile
    @timethis
    def rela_ori(self):
        # calculate relative orientation parameters
        xyf1 = xy_to_xyf(self.img_left, self.f, self.DIM)
        xyf2 = xy_to_xyf(self.img_right, self.f, self.DIM)
        # modify the drop index for your own data
        # xyf1,xyf2 = drop(xyf1, xyf2, [3, 14, 24])
        iterator = 0
        undropped = True
        drop_index = []
        while undropped:
            while True:
                R1 = rotation_matrix(
                    self.angle[0, 0], self.angle[0, 1], self.angle[0, 2])
                R2 = rotation_matrix(
                    self.angle[0, 3], self.angle[0, 4], self.angle[0, 5])
                XYZ1 = self.img_space_coord_to_assisted_img_space_coord(
                    xyf1, R1)
                XYZ2 = self.img_space_coord_to_assisted_img_space_coord(
                    xyf2, R2)
                A, L = self.XYZf_to_AL(XYZ1, XYZ2, self.f)
                delta_angle = or_lstsq(A, L)
                self.angle = np.expand_dims(
                    np.concatenate((self.angle[:, 0]+delta_angle[:, 0],
                                    self.angle[:, 1],
                                    self.angle[0, 2:6]+delta_angle[0, 1:5])), axis=0)
                iterator += 1

                if abs(delta_angle[0, 0]) < LIM_ERR and abs(delta_angle[0, 1]) < LIM_ERR and abs(delta_angle[0, 2]) < LIM_ERR and abs(delta_angle[0, 3]) < LIM_ERR and abs(delta_angle[0, 4]) < LIM_ERR:
                    break

            self.q = cal_q(self.f,
                           self.img_space_coord_to_assisted_img_space_coord(xyf1, R1)[
                               1, :],
                           self.img_space_coord_to_assisted_img_space_coord(xyf1, R1)[
                               2, :],
                           self.img_space_coord_to_assisted_img_space_coord(xyf2, R2)[
                               1, :],
                           self.img_space_coord_to_assisted_img_space_coord(xyf2, R2)[2, :])

            # automatically drop points for which q larger than 0.015
            for i in range(len(self.q)):
                if abs(self.q[i]) > 0.015:
                    drop_index.append(i)
            if drop_index:
                xyf1, xyf2 = drop(xyf1, xyf2, drop_index)
                drop_index.clear()
                iterator = 0
            else:
                undropped = False

        print("rela_orientation recycle times:", iterator)
        return self.angle, self.q, R1, R2
