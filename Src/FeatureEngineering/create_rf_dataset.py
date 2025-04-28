# @Author  : ChaoQiezi
# @Time    : 2025/4/25 上午11:29
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: create_rf_dataset

"""
This script is used to 用于创建用于随机森林(RF)模型训练和测试的样本
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import xarray as xr
import calendar
import rasterio as rio
from rasterio.plot import show

from Src.utils import extract1d_lons_lats


# 准备
aspect_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Aspect"
dem_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM"
evi_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\EVI"
prcp_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG"
lon_lat_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\LonLat"
lst_day_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\LST_Day"
lst_night_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\LST_Night"
ndvi_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\NDVI"
slope_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Slope"
ws_angle_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\wind_slope_angle"
prcp_class_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\prcp_class"
# res_name = '1km'
# sample_name = 'test'
res_name = '0.1deg'
sample_name = 'train'
var_names = ['dem', 'aspect', 'slope', 'lon', 'lat', 'ws_angle', 'ndvi', 'evi', 'lst_day', 'lst_night', 'prcp_class']
if res_name == '0.1deg':
    var_names.append('prcp')
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
rd = relativedelta(end_date, start_date)
months_count = rd.years * 12 + rd.months + 1  # 包括start_date和end_date所在月份(左右均闭区间)
out_path = r'E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_{}_samples.nc'.format(sample_name)  # 输出为nc格式
# 静态因子路径
dem_path = os.path.join(dem_dir, 'sw_china_dem_{}.tif'.format(res_name))
aspect_path = os.path.join(aspect_dir, 'aspect_{}.tif'.format(res_name))
slope_path = os.path.join(slope_dir, 'slope_{}.tif'.format(res_name))
lon_path = os.path.join(lon_lat_dir, 'lon_{}.tif'.format(res_name))
lat_path = os.path.join(lon_lat_dir, 'lat_{}.tif'.format(res_name))

# 迭代前预准备
lons, lats = extract1d_lons_lats(dem_path)
date_coords = pd.date_range(start_date, end_date, freq='MS')  # MS表示month start
da = xr.DataArray(
        dims=['date', 'var', 'lat', 'lon'],
        coords={
            'date': date_coords,
            'var': var_names,
            'lat': lats[::-1],  # 数组从上往下第一行区域是纬度最高的地方, 所以纬度要从大到小逆序传入!
            'lon': lons,
        },
        name=sample_name
    )
# 迭代读取存储
for cur_months_ix, cur_months in enumerate(range(months_count)):
    # 日期
    cur_date = start_date + relativedelta(months=cur_months)
    cur_year, cur_month =cur_date.year, cur_date.month
    # 路径
    ws_angle_path = os.path.join(ws_angle_dir, res_name, 'ws_angle_{}{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    ndvi_path = os.path.join(ndvi_dir, res_name, 'MOD13A3_NDVI_{}_{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    evi_path = os.path.join(evi_dir, res_name, 'MOD13A3_EVI_{}_{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    lst_day_path = os.path.join(lst_day_dir, res_name, 'MOD11A2_LST_Day_{}_{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    lst_night_path = os.path.join(lst_night_dir, res_name, 'MOD11A2_LST_Night_{}_{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    prcp_path = os.path.join(prcp_dir, res_name, 'GPM_{}_{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    prcp_class_path = os.path.join(prcp_class_dir, res_name, 'prcp_class_{}_{:02}_{}.tif'.format(cur_year, cur_month, res_name))
    # 创建样本
    paths = [dem_path, aspect_path, slope_path, lon_path, lat_path, ws_angle_path, ndvi_path, evi_path, lst_day_path,
             lst_night_path, prcp_class_path]
    if res_name == '0.1deg':
        paths.append(prcp_path)
    for cur_var_ix, (var_name, var_path) in enumerate(zip(var_names, paths)):
        with rio.open(var_path) as src:
            var = src.read(1, masked=True)
        da.loc[{'date': cur_date, 'var': var_name}] = var
        print('处理: {}{:02}-{}'.format(cur_year, cur_month, var_name))
# 输出为nc文件
da.to_netcdf(
    out_path,
    format='NETCDF4',
)
print('处理完成.')

