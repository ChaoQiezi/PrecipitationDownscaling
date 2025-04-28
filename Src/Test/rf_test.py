# @Author  : ChaoQiezi
# @Time    : 2025/4/28 下午2:01
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: rf_test

"""
This script is used to 
"""

import os
import joblib  # 保存模型
import numpy as np
import xarray as xr

import Config
from Src.utils import write_tiff


# 准备
train_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_test_samples.nc"
out_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction"
dem_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif"

# 预测(针对测试集)
da = xr.open_dataarray(train_path)
y_pred_reconstructed = xr.full_like(da.isel(var=0).drop_vars('var'), np.nan)

# 去除prcp_class分级
cur_da = da.drop_sel(var='prcp_class')
# 整理shape
cur_da = cur_da.stack(sample=['date', 'lat', 'lon']).transpose('sample', 'var')
cur_da = cur_da.dropna(dim='sample', how='any')
# 创建X
x = cur_da.values

# 模型预测
cur_model_path = os.path.join(Config.model_dir, 'rf.pkl')
rf = joblib.load(cur_model_path)

# 将预测值恢复原形状
samples = cur_da['sample'].to_index()
dates = samples.get_level_values('date')
lats = samples.get_level_values('lat')
lons = samples.get_level_values('lon')
date_ix = y_pred_reconstructed.date.to_index().get_indexer(dates)
lat_ix = y_pred_reconstructed.lat.to_index().get_indexer(lats)
lon_ix = y_pred_reconstructed.lon.to_index().get_indexer(lons)
y_pred = rf.predict(x)
y_pred_reconstructed.values[date_ix, lat_ix, lon_ix] = y_pred

# 输出预测结果(.nc)
out_nc_path = os.path.join(out_dir, 'rf_test_pred.nc')
y_pred_reconstructed.to_netcdf(out_nc_path)
# 输出预测结果(.tif)
y_pred_reconstructed = xr.open_dataarray(out_nc_path)
for date_step in range(len(y_pred_reconstructed)):
    cur_img = y_pred_reconstructed[date_step, :, :]
    cur_date = cur_img.date.values.astype('datetime64[D]').item()
    cur_out_filename = 'prcp_pred_{}_{:02}.tif'.format(cur_date.year, cur_date.month)
    cur_out_path = os.path.join(out_dir, cur_out_filename)
    write_tiff(cur_img.values, cur_out_path, dem_path, nodata_value=np.nan)
    print('输出: {}'.format(cur_out_filename))