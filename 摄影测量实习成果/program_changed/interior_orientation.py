import numpy as np
from utils import(
    or_lstsq,
    timethis,
    profile,
)


class Interior_Orientation:
    def __init__(self, img_x, img_y, pixel_x, pixel_y) -> None:
        self.img_x = img_x
        self.img_y = img_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def cal_coords(self) -> None:
        # take counterclockwise as positive direction
        self.img_coords_x = np.array(
            [[-self.img_x/2, -self.img_x/2, self.img_x/2, self.img_x/2]], dtype=np.float64)
        self.img_coords_y = np.array(
            [[self.img_y/2, -self.img_y/2, -self.img_y/2, self.img_y/2]], dtype=np.float64)
        self.pixel_coords_x = np.array(
            [[0, 0, self.pixel_x, self.pixel_x]], dtype=np.float64)
        self.pixel_coords_y = np.array(
            [[0, self.pixel_y, self.pixel_y, 0]], dtype=np.float64)

    @profile
    @timethis
    def inter_ori(self):
        # calculate interior orientation parameters
        self.cal_coords()
        L = np.append(self.img_coords_x, self.img_coords_y, axis=0).T
        A = np.concatenate(
            (np.ones((1, 4)), self.pixel_coords_x, self.pixel_coords_y), axis=0).T
        inter_ori_parameters = or_lstsq(A, L)
        self.inter_ori_parameters = inter_ori_parameters
        return self.inter_ori_parameters

    def pixel_to_img(self, pixel):
        return np.dot(self.inter_ori_parameters, pixel)
