# @Author  : ChaoQiezi
# @Time    : 2025/4/5 下午8:44
# @Email   : chaoqiezi.one@qq.com
# @FileName: gpm_preprocessing

"""
This script is used to 将GPM降水数据输出为Geotiff文件
"""

import os
import h5py
import numpy as np
from datetime import datetime, timezone
from dateutil import relativedelta
from dateutil.relativedelta import relativedelta
from osgeo import gdal, osr
from glob import glob

import Config
from Src.utils import hdf2tiff, clip_mask
gdal.UseExceptions()  # 使用异常错误机制(存在错误即报错而不是继续往下执行后续代码)


# 准备
gpm_dir = r'G:\GPM_IMERG_V07\Final_Month'
out_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG'
res_folder_name = '0.1deg'
out_res = 0.1
out_dir = os.path.join(out_dir, res_folder_name)
if not os.path.exists(out_dir):  # 不存在则创建文件夹
    os.makedirs(out_dir)

# 检索和迭代处理
gpm_paths = glob(os.path.join(gpm_dir, '3B-MO.MS.MRG.3IMERG.*.HDF5'))  # 迭代获取原始的GPM降水数据集(.HDF5文件)
for hdf_path in gpm_paths:
    hdf2tiff(hdf_path, out_dir)
    print('输出Geotiff成功: {}'.format(os.path.basename(hdf_path)))
dealt_tiff_paths = glob(os.path.join(out_dir, 'GPM*.tif'))  # 迭代获取hdf2tiff函数输出的geotiff文件
for tiff_path in dealt_tiff_paths:
    tiff_name = os.path.basename(tiff_path).split('.')[0] + '_{}.tif'.format(res_folder_name)
    cur_out_path = os.path.join(out_dir, tiff_name)
    clip_mask(tiff_path, Config.region_path, cur_out_path, out_res=out_res, remove_src=True, resample_alg=0)
    """
    这里由于降水量采用双线性插值和三次卷积插值都可能会出现了负数,但是降水量不会是负数,因此这里使用最近邻
    """
    print('裁剪掩膜成功: {}'.format(os.path.basename(tiff_name)))
print('处理完成.')