from data_loading import _DATA
from interior_orientation import Interior_Orientation
from relative_orientation import Relative_Orientation
from forward_intersection import Forward_Intersection
from absolute_orientation import Absolute_Orientation
from utils import(
    write_file,
    or_read_excel,
    xy_to_xyf,
    timethis,
    profile,
    point_proj_coeffi,
    write_excel,
)
import numpy as np
import logging

DATA = _DATA()


@profile
@timethis
def main():
    # interior orientation
    interior_orientation = Interior_Orientation(
        DATA.IMG_X, DATA.IMG_Y, DATA.PIXEL_X, DATA.PIXEL_Y)
    inter_ori_parameters = interior_orientation.inter_ori()
    # save inter_ori_parameters to folder
    
    write_file(str(inter_ori_parameters),
               DATA.SAVE_IOP_FILE_NAME, DATA.SAVE_DIR)
    

    # relative orientation
    rops = or_read_excel(DATA.FILE_DIR, DATA.READ_ROPS_FILE_NAME)
    rel_pixel_left, rel_pixel_right = rops[0:2, :], rops[2:4, :]
    rel_img_left = interior_orientation.pixel_to_img(
        np.concatenate((np.ones((1, rops.shape[1])), rel_pixel_left), axis=0))
    rel_img_right = interior_orientation.pixel_to_img(
        np.concatenate((np.ones((1, rops.shape[1])), rel_pixel_right), axis=0))

    relative_orientation = Relative_Orientation(
        rel_img_left, rel_img_right, DATA.F, rops.shape[1])
    rela_ori_parameters, q, R1, R2 = relative_orientation.rela_ori()
    
    print ("q\n",q)
    # save inter_ori_parameters to folder
    write_file(str(rela_ori_parameters),
               DATA.SAVE_ROP_FILE_NAME, DATA.SAVE_DIR)
    

    # forward intersection
    """
    B = l * (1 - p) * M
    use all ground points (they are corresponding image points samely) to calculate p and M
    """
    gps = or_read_excel(DATA.FILE_DIR, DATA.READ_GPS_FILE_NAME)
    gps_pixel_left, gps_pixel_right, ground_points = gps[0:2,
                                                         :], gps[2:4, :], gps[4:7, :]
    gps_img_left = interior_orientation.pixel_to_img(
        np.concatenate((np.ones((1, gps.shape[1])), gps_pixel_left), axis=0))
    gps_img_right = interior_orientation.pixel_to_img(
        np.concatenate((np.ones((1, gps.shape[1])), gps_pixel_right), axis=0))

    gps_assisted_img_left = relative_orientation.img_space_coord_to_assisted_img_space_coord(
        xy_to_xyf(gps_img_left, DATA.F, gps.shape[1]), R1
    )
    gps_assisted_img_right = relative_orientation.img_space_coord_to_assisted_img_space_coord(
        xy_to_xyf(gps_img_right, DATA.F, gps.shape[1]), R2
    )

    forward_intersection = Forward_Intersection(
        gps_img_left, gps_img_right, ground_points,
        gps_assisted_img_left, gps_assisted_img_right,
        DATA.L, gps.shape[1], rela_ori_parameters
    )
    # measured in mm
    B, Q, gps_model = forward_intersection.forw_intersect()
    
    print("Q\n",Q)
    # print("gps_model\n",gps_model)
    # save gps_model to folder
    write_file(str(gps_model),
               DATA.SAVE_GPS_MODEL_FILE_NAME, DATA.SAVE_DIR)
    

    # absolute orientation
    # from here we use meter as measurement
    gps_model = gps_model/1000
    absolute_orientation = Absolute_Orientation(
        gps_model, ground_points
    )
    ab_ori_parameters, gps_ground, error = absolute_orientation.ab_orientation()
    
    print("error\n",error)
    # save ab_ori_parameters to folder
    write_file(str(ab_ori_parameters),
               DATA.SAVE_AOP_FILE_NAME, DATA.SAVE_DIR)
    


    # use image pixel coordinates for unknown points to calculate its actual positions in ground coordinates
    # calculate image fiducial mark coordinates
    pfms = or_read_excel(DATA.FILE_DIR, DATA.READ_PFM_FILE_NAME)
    pfms_pixel_left, pfms_pixel_right = pfms[0:2, :], pfms[2:4, :]
    pfms_img_left = interior_orientation.pixel_to_img(
        np.concatenate((np.ones((1, pfms.shape[1])), pfms_pixel_left), axis=0))
    pfms_img_right = interior_orientation.pixel_to_img(
        np.concatenate((np.ones((1, pfms.shape[1])), pfms_pixel_right), axis=0))

    # calculate model coordinates
    pfms_assisted_img_left = relative_orientation.img_space_coord_to_assisted_img_space_coord(
        xy_to_xyf(pfms_img_left, DATA.F, pfms.shape[1]), R1
    )
    pfms_assisted_img_right = relative_orientation.img_space_coord_to_assisted_img_space_coord(
        xy_to_xyf(pfms_img_right, DATA.F, pfms.shape[1]), R2
    )
    N_left, N_right = point_proj_coeffi(
        pfms_assisted_img_left[0,
                               :], pfms_assisted_img_right[0, :],
        pfms_assisted_img_left[2,
                               :], pfms_assisted_img_right[2, :],
        B[0], B[2])
    pfms_model = forward_intersection.cal_model_coord(
        B, N_left, N_right, pfms_assisted_img_left, pfms_assisted_img_right
    )

    # save pfms_model to folder
    write_file(str(pfms_model),
               DATA.SAVE_PFMS_MODEL_FILE_NAME, DATA.SAVE_DIR)

    # calculate ground coordinates
    pfms_model = pfms_model/1000
    pfms_ground = absolute_orientation.cal_ground_coord(
        ab_ori_parameters, pfms_model)

    # write your results to the same input excel 
    
    write_excel(DATA.FILE_DIR, DATA.READ_PFM_FILE_NAME, pfms_ground)
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.log(10, "Exception Reason:", e)
