# @Author  : ChaoQiezi
# @Time    : 2025/5/13 上午5:19
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: cal_mean

"""
This script is used to 用于计算预测结果的季均值、年均值、月均值
"""

import os
import numpy as np
import xarray as xr
from rasterio.plot import show

from Src.utils import write_tiff, mean_monthly_seasonally_yearly, mask_metric, extract1d_lons_lats


# 准备
prcp_1km_pred_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf_test_pred_by_residual.nc"
train_path = r'E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_train_samples.nc'
out_1km_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf'
out_0p1deg_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG\0.1deg'
template_1km_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif"  # 作为写入tiff文件的参考地理文件使用
template_0p1deg_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_0.1deg.tif"  # 作为写入tiff文件的参考地理文件使用
temp_path = r'E:\MyTEMP\temp.tif'
mask_dir = r'E:\SoftwaresStorage\ArcGISPro_Storage\Projects\PrecipitationDownscaling\research_region'
mask_filenames = ['Sichuan.shp', 'Chongqing.shp', 'Yunnan.shp', 'Guizhou.shp']

# 计算1km降水的月、季、年
prcp_1km_pred = xr.open_dataarray(prcp_1km_pred_path)
mean_monthly_seasonally_yearly(prcp_1km_pred, out_1km_dir, template_1km_path)
# 掩膜各种尺度的月、季、年
# 掩膜月
prcp_month_mean_path = os.path.join(out_1km_dir, 'monthly', 'prcp_month_mean.nc')
prcp_month_mean = xr.open_dataarray(prcp_month_mean_path)
for cur_filename in mask_filenames:
    masked_metric_list = []
    cur_region_name = cur_filename.split('.')[0]
    mask_path = os.path.join(mask_dir, cur_filename)
    for date_ix in prcp_month_mean.month.values:
        cur_metric = prcp_month_mean.loc[date_ix, :, :].values
        masked_img = mask_metric(cur_metric, template_1km_path, mask_path, out_path=temp_path)
        masked_metric_list.append(masked_img.GetRasterBand(1).ReadAsArray())
        masked_img = None
    lons, lats = extract1d_lons_lats(temp_path)
    da = xr.DataArray(
            data=masked_metric_list,
            dims=['month', 'lat', 'lon'],
            coords={
                'month': prcp_month_mean.month.values,
                'lat': lats[::-1],  # 数组从上往下第一行区域是纬度最高的地方, 所以纬度要从大到小逆序传入!
                'lon': lons,
            },
            name='month_{}'.format(cur_region_name)
        )
    out_path = os.path.join(out_1km_dir, 'monthly', 'prcp_month_mean_{}.nc'.format(cur_region_name))
    da.to_netcdf(out_path)
    print('掩膜处理(月): {}'.format(cur_filename))
# 掩膜季
prcp_season_mean_path = os.path.join(out_1km_dir, 'seasonally', 'prcp_season_mean.nc')
prcp_season_mean = xr.open_dataarray(prcp_season_mean_path)
for cur_filename in mask_filenames:
    masked_metric_list = []
    cur_region_name = cur_filename.split('.')[0]
    mask_path = os.path.join(mask_dir, cur_filename)
    for date_ix in prcp_season_mean.season.values:
        cur_metric = prcp_season_mean.loc[date_ix, :, :].values
        masked_img = mask_metric(cur_metric, template_1km_path, mask_path, out_path=temp_path)
        masked_metric_list.append(masked_img.GetRasterBand(1).ReadAsArray())
        masked_img = None
    lons, lats = extract1d_lons_lats(temp_path)
    da = xr.DataArray(
            data=masked_metric_list,
            dims=['season', 'lat', 'lon'],
            coords={
                'season': prcp_season_mean.season.values,
                'lat': lats[::-1],  # 数组从上往下第一行区域是纬度最高的地方, 所以纬度要从大到小逆序传入!
                'lon': lons,
            },
            name='season_{}'.format(cur_region_name)
        )
    out_path = os.path.join(out_1km_dir, 'seasonally', 'prcp_season_mean_{}.nc'.format(cur_region_name))
    da.to_netcdf(out_path)
    print('掩膜处理(季): {}'.format(cur_filename))
# 掩膜年
prcp_year_mean_path = os.path.join(out_1km_dir, 'yearly', 'prcp_year_mean.nc')
prcp_year_mean = xr.open_dataarray(prcp_year_mean_path)
for cur_filename in mask_filenames:
    masked_metric_list = []
    cur_region_name = cur_filename.split('.')[0]
    mask_path = os.path.join(mask_dir, cur_filename)
    for date_ix in range(2019, 2024):
        cur_metric = prcp_year_mean.loc[date_ix, :, :].values
        masked_img = mask_metric(cur_metric, template_1km_path, mask_path, out_path=temp_path)
        masked_metric_list.append(masked_img.GetRasterBand(1).ReadAsArray())
        masked_img = None
    lons, lats = extract1d_lons_lats(temp_path)
    da = xr.DataArray(
            data=masked_metric_list,
            dims=['year', 'lat', 'lon'],
            coords={
                'year': np.arange(2019, 2024),
                'lat': lats[::-1],  # 数组从上往下第一行区域是纬度最高的地方, 所以纬度要从大到小逆序传入!
                'lon': lons,
            },
            name='year_{}'.format(cur_region_name)
        )
    out_path = os.path.join(out_1km_dir, 'yearly', 'prcp_year_mean_{}.nc'.format(cur_region_name))
    da.to_netcdf(out_path)
    print('掩膜处理(年): {}'.format(cur_filename))

# 计算0.1°降水的月、季、年
prcp_0p1deg_pred = xr.open_dataarray(train_path).sel(var='prcp').drop_vars('var')
mean_monthly_seasonally_yearly(prcp_0p1deg_pred, out_0p1deg_dir, template_0p1deg_path)
# 掩膜年
prcp_year_mean_path = os.path.join(out_0p1deg_dir, 'yearly', 'prcp_year_mean.nc')
prcp_year_mean = xr.open_dataarray(prcp_year_mean_path)
for cur_filename in mask_filenames:
    masked_metric_list = []
    cur_region_name = cur_filename.split('.')[0]
    mask_path = os.path.join(mask_dir, cur_filename)
    for date_ix in range(2019, 2024):
        cur_metric = prcp_year_mean.loc[date_ix, :, :].values
        masked_img = mask_metric(cur_metric, template_0p1deg_path, mask_path, out_path=temp_path)
        masked_metric_list.append(masked_img.GetRasterBand(1).ReadAsArray())
        masked_img = None
    lons, lats = extract1d_lons_lats(temp_path)
    da = xr.DataArray(
            data=masked_metric_list,
            dims=['year', 'lat', 'lon'],
            coords={
                'year': np.arange(2019, 2024),
                'lat': lats[::-1],  # 数组从上往下第一行区域是纬度最高的地方, 所以纬度要从大到小逆序传入!
                'lon': lons,
            },
            name='year_{}'.format(cur_region_name)
        )
    out_path = os.path.join(out_0p1deg_dir, 'yearly', 'prcp_year_mean_{}.nc'.format(cur_region_name))
    da.to_netcdf(out_path)
    print('掩膜处理(年): {}'.format(cur_filename))