import numpy as np
from data_loading import _DATA
from utils import(
    simp_rota_matrix,
    or_lstsq,
    timethis,
    profile,
    drop,
)

LIM_ERR_XYZ = _DATA.LIM_ERR_AO_XYZ
LIM_ERR_ANG = _DATA.LIM_ERR_AO_ANG


class Absolute_Orientation:
    def __init__(self, gps_model, ground_points) -> None:
        self.gps_model = gps_model
        self.ground_points = ground_points
        self.DIM = self.gps_model.shape[1]
        self.ab_ori_parameters = np.expand_dims(
            np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float64), axis=0)

    def cal_center_coord(self, Reference, Target):
        return Target-np.dot(np.expand_dims(np.mean(Reference, axis=1), axis=0).T, np.ones((1, Target.shape[1])))

    def pts_to_AL(self, XYZ_model, XYZ_points, ab_ori_parameters, R, DIM):
        # A and L both have 3*DIM rows, but deltax, deltay and deltaz are unique
        # coefficients are aligned by Xis, Yis and Zis
        X_m = np.expand_dims(XYZ_model[0, :], axis=0).T
        Y_m = np.expand_dims(XYZ_model[1, :], axis=0).T
        Z_m = np.expand_dims(XYZ_model[2, :], axis=0).T

        A1 = np.concatenate((np.ones((DIM, 1)), np.zeros(
            (DIM, 2)), X_m, np.zeros((DIM, 1)), -Z_m, Y_m), axis=1)
        A2 = np.concatenate((np.zeros((DIM, 1)), np.ones((DIM, 1)), np.zeros(
            (DIM, 1)), Y_m, Z_m, np.zeros((DIM, 1)), -X_m), axis=1)
        A3 = np.concatenate((np.zeros((DIM, 2)), np.ones(
            (DIM, 1)), Z_m, -Y_m, X_m, np.zeros((DIM, 1))), axis=1)
        A = np.concatenate((A1, A2, A3), axis=0)

        # as L uses plus and minus mainly, change its shape to 3*DIM rows after calculation
        L0 = XYZ_points-ab_ori_parameters[0, 3] * np.dot(R, XYZ_model)-np.dot(
            np.expand_dims(ab_ori_parameters[0, 0:3], axis=0).T, np.ones((1, DIM)))
        L1 = np.expand_dims(L0[0, :], axis=0).T
        L2 = np.expand_dims(L0[1, :], axis=0).T
        L3 = np.expand_dims(L0[2, :], axis=0).T
        L = np.concatenate((L1, L2, L3), axis=0)
        return A, L

    def cal_ground_coord(self, ab_ori_parameters, model_coord):
        cen_model_coord = self.cal_center_coord(self.gps_model, model_coord)
        R = simp_rota_matrix(
            ab_ori_parameters[0, 4], ab_ori_parameters[0, 5], ab_ori_parameters[0, 6])
        ground_coord = ab_ori_parameters[0, 3]*np.dot(R, cen_model_coord)+np.dot(
            np.expand_dims(
                ab_ori_parameters[0, 0:3]+np.mean(self.ground_points, axis=1), axis=0).T,
            np.ones((1, model_coord.shape[1])))
        # ground_coord[[0, 1], :] = ground_coord[[1, 0], :]
        return ground_coord

    @profile
    @timethis
    def ab_orientation(self):
        self.ground_points[[0, 1], :] = self.ground_points[[1, 0], :]
        iterator = 0
        undropped = True
        drop_index = []
        while undropped:
            self.cen_gps_model = self.cal_center_coord(
                self.gps_model, self.gps_model)
            self.cen_ground_points = self.cal_center_coord(
                self.ground_points, self.ground_points)
            while True:
                R = simp_rota_matrix(
                    self.ab_ori_parameters[0, 4], self.ab_ori_parameters[0, 5], self.ab_ori_parameters[0, 6])
                A, L = self.pts_to_AL(
                    self.cen_gps_model, self.cen_ground_points, self.ab_ori_parameters, R, self.DIM)
                delta_aop = or_lstsq(A, L)
                self.ab_ori_parameters += delta_aop
                iterator += 1
                if abs(delta_aop[0, 0]) < LIM_ERR_XYZ and abs(delta_aop[0, 1]) < LIM_ERR_XYZ and abs(delta_aop[0, 2]) < LIM_ERR_XYZ and abs(delta_aop[0, 3]) < LIM_ERR_ANG and abs(delta_aop[0, 4]) < LIM_ERR_ANG and abs(delta_aop[0, 5]) < LIM_ERR_ANG and abs(delta_aop[0, 6]) < LIM_ERR_ANG:
                    break

            self.gps_ground = self.cal_ground_coord(
                self.ab_ori_parameters, self.gps_model)
            self.error = self.ground_points-self.gps_ground
            # automatically drop points for which error larger than 1m
            for i in range(self.error.shape[1]):
                if abs(self.error[0, i]) > 1 or abs(self.error[1, i]) > 1:
                    drop_index.append(i)
            if drop_index:
                self.gps_model, self.ground_points = drop(
                    self.gps_model, self.ground_points, drop_index)
                self.DIM = self.gps_model.shape[1]
                drop_index.clear()
                iterator = 0
            else:
                undropped = False

        print("ab_orientation recycle times:", iterator)
        return self.ab_ori_parameters, self.gps_ground, self.error
