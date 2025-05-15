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
import pandas as pd
import xarray as xr
from osgeo import gdal
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt
from dateutil.relativedelta import relativedelta
from rasterio.plot import show

import Config
from Src.utils import write_tiff, predict_model, resample_metric


# 准备
test_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_test_samples.nc"
train_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_train_samples.nc"
out_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction"
prcp_st_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Station\prcp_st_month_sum.xlsx"
dem_1km_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif"  # 作为写入tiff文件的参考地理文件使用
dem_0p1deg_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_0.1deg.tif"  # 作为写入tiff文件的参考地理文件使用

# 读取
prcp_st = pd.read_excel(prcp_st_path)
prcp_st['date'] = pd.to_datetime(prcp_st['YM'], format='%Y/%m/%d')
prcp_st = prcp_st[(prcp_st['date'] >= Config.start_date) & (prcp_st['date'] <= Config.end_date)]
prcp_st['prcp_pred'] = np.nan

"""
预测不是简单的预测,遵照下面的步骤进行

1. 将 0.1°解释变量输入到步骤(3)的模型中,得到 0.1°分辨率的模型预测值；
2. 计算原始遥感降水产品与 0.1°模型预测值之间的残差,并将残差值插值  为 0.01°分辨率;
3. 将 0.01°的解释变量输入到步骤(3)的模型中,得到 0.01°分辨率的模型预  测值。
    并将其与步骤(5) 0.01°残差数据相加,得到最终的 0.01°高分辨率的遥感降尺度结果。
"""
# 加载模型
cur_model_path = os.path.join(Config.model_dir, 'rf.pkl')
rf = joblib.load(cur_model_path)
# 预测1km下的降水值
out_nc_path = os.path.join(out_dir, 'rf_test_pred.nc')
predict_model(rf, test_path, out_nc_path, flag='test')
prcp_1km_pred_da = xr.open_dataarray(out_nc_path).load()
# 预测0.1°下的降水值
out_nc_path = os.path.join(out_dir, 'rf_train_pred.nc')
predict_model(rf, train_path, out_nc_path)
prcp_0p1deg_pred_da = xr.open_dataarray(out_nc_path)
# 计算0.1°下的残差
prcp_0p1deg_da = xr.open_dataarray(train_path).sel(var='prcp').drop_vars('var')
residual_0p1deg_da = prcp_0p1deg_da - prcp_0p1deg_pred_da
# 0.1°残差重采样为1km
residual_1km_da = xr.full_like(prcp_1km_pred_da, np.nan)  # 创建跟da一样shape的数组
for ix in range(len(residual_0p1deg_da)):
    cur_residual_0p1deg = residual_0p1deg_da[ix, :, :]
    cur_residual_1km = resample_metric(cur_residual_0p1deg.values, dem_0p1deg_path)
    residual_1km_da[ix, :, :] = cur_residual_1km
# 残差叠加
prcp_1km_pred_da = prcp_1km_pred_da + residual_1km_da
prcp_1km_pred_da = xr.where(prcp_1km_pred_da < 0, 0, prcp_1km_pred_da)  # 剔除小于0的降水量
# 输出叠加结果(.nc)
out_nc_path = os.path.join(out_dir, 'rf_test_pred_by_residual.nc')
prcp_1km_pred_da.to_netcdf(out_nc_path)

# 输出预测结果(.tif)
for date_step in range(len(prcp_1km_pred_da)):
    cur_img = prcp_1km_pred_da[date_step, :, :]
    cur_date = Config.start_date + relativedelta(months=date_step)
    cur_prcp_st = prcp_st[prcp_st['date'] == cur_date]
    for cur_ix, cur_row in cur_prcp_st.iterrows():
        cur_lat, cur_lon = cur_row.Lat, cur_row.Lon
        prcp_st.loc[cur_ix, 'prcp_pred'] = cur_img.sel(lat=cur_lat, lon=cur_lon, method='nearest', tolerance=0.1)  # 设置最大捕获距离为0.1°
    cur_date = cur_img.date.values.astype('datetime64[D]').item()
    cur_out_filename = 'prcp_pred_{}_{:02}.tif'.format(cur_date.year, cur_date.month)
    cur_out_path = os.path.join(out_dir, 'rf', cur_out_filename)
    write_tiff(cur_img.values, cur_out_path, dem_1km_path, nodata_value=np.nan)
    print('输出: {}'.format(cur_out_filename))

    cur_df = prcp_st[prcp_st['date'] == cur_date].dropna(subset=['prcp', 'prcp_pred'])

# 输出长格式(方便统计分析)
prcp_st.to_excel(os.path.join(out_dir, 'table', 'pred_prcp_st.xlsx'), index=False)
# 长格式转宽格式(方便转shp)
prcp_st_melt = prcp_st.pivot(index=['station_id', 'Lat', 'Lon'], columns='date', values=['prcp', 'prcp_pred'])
prcp_st_melt.columns = [_col[1].strftime('%Y-%m-%d') + '_' + _col[0] for _col in prcp_st_melt.columns]  # 多级索引转普通索引
prcp_st_melt = prcp_st_melt.reset_index()
prcp_st_melt.to_excel(os.path.join(out_dir, 'table', 'pred_prcp_st_melt.xlsx'), index=False)
