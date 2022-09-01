from functools import wraps


def Const(cls):
    @wraps(cls)
    def new_setattr(self, name, value):
        raise Exception('const : {} cannot be changed'.format(name))

    cls.__setattr__ = new_setattr
    return cls


@Const
class _DATA(object):
    # change your data here
    IMG_X = 56.90880000002289
    IMG_Y = 100.3392000000404
    PIXEL_X = 14592
    PIXEL_Y = 25728
    F = 92
    SAVE_DIR = "C:/Users/gao_yang/Desktop/摄影测量实习/1951726高扬_摄影测量实习成果/program_changed"
    FILE_DIR = "C:/Users/gao_yang/Desktop/摄影测量实习/1951726高扬_摄影测量实习成果/program_changed"
    SAVE_IOP_FILE_NAME = "inter_ori_parameters"
    SAVE_ROP_FILE_NAME = "rela_ori_parameters"
    SAVE_GPS_MODEL_FILE_NAME = "gps_model"
    SAVE_PFMS_MODEL_FILE_NAME = "pfms_model"
    SAVE_AOP_FILE_NAME = "ab_ori_parameters"
    READ_ROPS_FILE_NAME = "relative_orientation_points.xlsx"
    READ_GPS_FILE_NAME = "gp_recovered.xlsx"
    L = IMG_X
    # limited error in relative orientation
    LIM_ERR_RO = 0.0003
    # limited error in absolute orientation
    LIM_ERR_AO_XYZ = 0.1
    LIM_ERR_AO_ANG = 0.000001


    # input your points for mapping here
    READ_PFM_FILE_NAME = "all_points_for_mapping.xlsx"
    