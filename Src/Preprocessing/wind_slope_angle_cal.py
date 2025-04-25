# @Author  : ChaoQiezi
# @Time    : 2025/4/21 下午10:12
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: wind_slope_angle_cal

"""
This script is used to 基于ERA5的u和v风向计算风向角度(0~360),再利用风向和坡度坡向计算风坡夹角.

风坡夹角: 风向与坡向夹角的余弦值和坡度正弦值的乘积.
"""
import os.path

import numpy as np
from numpy import arctan2, pi
from datetime import datetime
from dateutil.relativedelta import relativedelta
from rasterio.plot import show

from Src.utils import cal_wind_direction, cal_wind_slope_angle

# 准备
out_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\wind_slope_angle'
slope_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Slope"
aspect_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Aspect"
u10_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\u10'
v10_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\v10'
res_folder_name = '1km'
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
out_dir = os.path.join(out_dir, res_folder_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

rd = relativedelta(end_date, start_date)
month_range = rd.years * 12 + rd.months + 1
for cur_month in range(month_range):
    cur_date = start_date + relativedelta(months=cur_month)
    cur_slope_path = os.path.join(slope_dir, 'slope_{}.tif'.format(res_folder_name))
    cur_aspect_path = os.path.join(aspect_dir, 'aspect_{}.tif'.format(res_folder_name))
    cur_u10_path = os.path.join(u10_dir, res_folder_name, 'u10_{}{:02}_{}.tif'.format(
        cur_date.year, cur_date.month, res_folder_name))
    cur_v10_path = os.path.join(v10_dir, res_folder_name, 'v10_{}{:02}_{}.tif'.format(
        cur_date.year, cur_date.month, res_folder_name))
    cur_out_filename = 'ws_angle_{}{:02}_{}.tif'.format(cur_date.year, cur_date.month, res_folder_name)
    cur_out_path = os.path.join(out_dir, cur_out_filename)
    cal_wind_slope_angle(cur_out_path, cur_u10_path, cur_v10_path, cur_aspect_path, cur_slope_path)
    print('处理: {}'.format(cur_out_filename))
print('处理完成.')
