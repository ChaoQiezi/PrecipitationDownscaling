# @Author  : ChaoQiezi
# @Time    : 2025/5/13 上午5:19
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: cal_mean

"""
This script is used to 用于计算预测结果的季均值、年均值、月均值
"""

import os
import xarray as xr

from Src.utils import write_tiff, mean_monthly_seasonally_yearly, mask_metric


# 准备
prcp_1km_pred_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf_test_pred_by_residual.nc"
train_path = r'E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_train_samples.nc'
out_1km_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf'
out_0p1deg_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG\0.1deg'
template_1km_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif"  # 作为写入tiff文件的参考地理文件使用
template_0p1deg_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_0.1deg.tif"  # 作为写入tiff文件的参考地理文件使用

# 计算1km降水的月、季、年
prcp_1km_pred = xr.open_dataarray(prcp_1km_pred_path)
mean_monthly_seasonally_yearly(prcp_1km_pred, out_1km_dir, template_1km_path)
# 掩膜各种尺度的月、季、年
# 掩膜月
prcp_month_mean_path = os.path.join(out_1km_dir, 'monthly', 'prcp_month_mean.nc')
prcp_month_mean = xr.open_dataarray(prcp_month_mean_path)
for date_ix in range(len(prcp_month_mean)):
    cur_metric = prcp_month_mean[date_ix, :, :].values
    mask_metric(cur_metric, template_1km_path, )
# 计算0.1°降水的月、季、年
prcp_0p1deg_pred = xr.open_dataarray(train_path).sel(var='prcp').drop_vars('var')
mean_monthly_seasonally_yearly(prcp_0p1deg_pred, out_0p1deg_dir, template_0p1deg_path)
