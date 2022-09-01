import numpy as np
from utils import(
    or_sqrt,
    timethis,
    profile,
    point_proj_coeffi,
    drop,
)


class Forward_Intersection:
    def __init__(
            self, gps_img_left, gps_img_right, ground_points,
            gps_assisted_img_left, gps_assisted_img_right,
            L, DIM, rela_ori_parameters) -> None:
        self.gps_img_left = gps_img_left
        self.gps_img_right = gps_img_right
        self.ground_points = ground_points
        self.gps_assisted_img_left = gps_assisted_img_left
        self.gps_assisted_img_right = gps_assisted_img_right
        self.L = L
        self.DIM = DIM
        self.rela_ori_parameters = rela_ori_parameters

    def cal_p(self):
        return 1-np.float64(np.sum(abs(self.gps_img_left[0, :]-self.gps_img_right[0, :]), axis=0)/(self.DIM*self.L))

    def cal_M(self):
        summary = np.float64(0)
        iterater = 0

        for i in range(self.DIM):
            for j in range(self.DIM-1, i, -1):
                img_dist_left = or_sqrt(
                    self.gps_img_left[:, i], self.gps_img_left[:, j])
                img_dist_right = or_sqrt(
                    self.gps_img_right[:, i], self.gps_img_right[:, j])
                # consider x and y
                gp_dist = or_sqrt(
                    self.ground_points[:, i], self.ground_points[:, j])
                s = np.float64((2000*gp_dist)/(img_dist_left+img_dist_right))

                summary += s
                iterater += 1

        M = np.float64(summary/iterater)
        return M

    def cal_B(self):
        B_ini = np.float64(self.L*(1-self.cal_p())*self.cal_M())
        return np.array([B_ini, 0, 0])

    def cal_model_coord(self, B, N_left, N_right, XYZ_left, XYZ_right):
        X_model = B[0]+N_right*XYZ_right[0, :]
        Y_model = (N_left*XYZ_left[1, :]+N_right*XYZ_right[1, :]+B[1])/2
        Z_model = B[2]+N_right*XYZ_right[2, :]
        return np.array([X_model, Y_model, Z_model])

    def cal_Q(self, N_left, N_right, XYZ_left, XYZ_right):
        return N_left*XYZ_left[1, :]-N_right*XYZ_right[1, :]

    @profile
    @timethis
    def forw_intersect(self):
        self.B = self.cal_B()
        self.N_left, self.N_right = point_proj_coeffi(
            self.gps_assisted_img_left[0,
                                       :], self.gps_assisted_img_right[0, :],
            self.gps_assisted_img_left[2,
                                       :], self.gps_assisted_img_right[2, :],
            self.B[0], self.B[2])
        # all variables are measured in mm, so the limitation of Q is 1000
        self.Q = self.cal_Q(self.N_left, self.N_right,
                            self.gps_assisted_img_left, self.gps_assisted_img_right)
        self.XYZ_model = self.cal_model_coord(
            self.B, self.N_left, self.N_right, self.gps_assisted_img_left, self.gps_assisted_img_right
        )
        return self.B, self.Q, self.XYZ_model
