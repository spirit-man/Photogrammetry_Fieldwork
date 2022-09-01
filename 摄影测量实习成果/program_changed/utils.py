import os
import numpy as np
import pandas as pd
import openpyxl
from math import sqrt
import time
from functools import wraps
from contextlib import contextmanager
from scipy import linalg
import psutil


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def profile(func):
    # memory usage for functions
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}.{} : consumed memory: {}bytes".format(
            func.__module__,
            func.__name__,
            mem_after - mem_before))
        return result
    return wrapper


def timethis(func):
    # running time for functions
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print("{}.{} : {}seconds".format(
            func.__module__, func.__name__, end - start))
        return r
    return wrapper


@contextmanager
def timeblock(label):
    # running time for blocks
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print("{} : {}seconds".format(label, end - start))


def write_file(str, file_name, save_dir):
    # write results to txt
    file_path = os.path.join(save_dir, file_name)+'.txt'
    f = open(file_path, "w")
    f.write(str)
    f.close()


def or_read_excel(file_dir, file_name):
    # return numpy array
    # your excel should start from cell B3 by default, override this function if not
    # read numbers only
    file_path = os.path.join(file_dir, file_name)
    data = np.float64(
        np.delete(np.array(pd.read_excel(file_path, header=1)), 0, axis=1).T)
    return data


def write_excel(file_dir, file_name, data):
    # write your results to the same excel as above
    # samely, your excel should start from cell B3 by default, override this function if not
    file_path = os.path.join(file_dir, file_name)
    diction = {"X": list(data[0, :]), "Y": list(
        data[1, :]), "Z": list(data[2, :])}
    df = pd.DataFrame(diction)
    df_ori = pd.DataFrame(pd.read_excel(file_path))
    book = openpyxl.load_workbook(file_path)
    df_cols = df_ori.shape[1]
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        df.to_excel(writer, startrow=1, startcol=df_cols +
                    1, index=False, header=True)


def rotation_matrix(fai, omiga, kappa):
    # from euler angle to rotation matrix
    r_fai = np.array([[np.cos(fai), 0., -np.sin(fai)],
                      [0., 1., 0.],
                      [np.sin(fai), 0., np.cos(fai)]], dtype=np.float64)

    r_omiga = np.array([[1., 0., 0.],
                        [0., np.cos(omiga), -np.sin(omiga)],
                        [0., np.sin(omiga), np.cos(omiga)]], dtype=np.float64)

    r_kappa = np.array([[np.cos(kappa), -np.sin(kappa), 0.],
                        [np.sin(kappa), np.cos(kappa), 0.],
                        [0., 0., 1.]], dtype=np.float64)

    R = np.dot(np.dot(r_fai, r_omiga), r_kappa)
    return R


def simp_rota_matrix(ex, ey, ez):
    # simplified rotation matrix
    simp_R = np.array([[1., ez, -ey],
                       [-ez, 1., ex],
                       [ey, -ex, 1.]], dtype=np.float64)
    return simp_R


def or_lstsq(A, L):
    # np.linalg.inv will significantly increase time expenses
    # return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), L).T
    return linalg.lstsq(A, L)[0].T


def point_proj_coeffi(X1, X2, Z1, Z2, BX, BZ):
    # calculate point projection coefficients
    left_proj_coeffi = (BX*Z2-BZ*X2)/(X1*Z2-Z1*X2)
    right_proj_coeffi = (BX*Z1-BZ*X1)/(X1*Z2-Z1*X2)
    return left_proj_coeffi, right_proj_coeffi


def xy_to_xyf(img, f, DIM):
    return np.append(img, f*np.ones((1, DIM)), axis=0)


def or_sqrt(xy1, xy2):
    # self defined sqrt function
    return np.float64(sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2))


def cal_q(f, Y1, Z1, Y2, Z2):
    # calculate vertical parallax
    return f*(Y1/Z1-Y2/Z2)


def drop(xyz1, xyz2, index):
    # delete incorrect corresponding points
    return np.delete(xyz1, index, axis=1), np.delete(xyz2, index, axis=1)
