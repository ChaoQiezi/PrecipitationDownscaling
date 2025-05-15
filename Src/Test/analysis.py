# @Author  : ChaoQiezi
# @Time    : 2025/5/11 下午11:02
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: analysis

"""
This script is used to 针对预测的结果进行精度评定分析
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from rasterio.plot import show
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

import Config
from Src.utils import extract_year_season


# 准备
train_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_train_samples.nc"
prcp_1km_pred_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf_test_pred_by_residual.nc"
prcp_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\table\pred_prcp_st.xlsx"
out_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Result\table"
# 读取和预处理
da = xr.open_dataarray(train_path).sel(var='prcp').drop_vars('var')  # 读取0.1deg的训练集的降水数据
prcp_df = pd.read_excel(prcp_path)  # 读取包含各个年份1km和站点
prcp_df = prcp_df[(prcp_df['date'] >= Config.start_date) & (prcp_df['date'] <= Config.end_date)]  #提取指定时间范围的df
prcp_df.rename(columns={'prcp_pred': 'prcp_1km', 'prcp': 'prcp_st'}, inplace=True)  # 更换列名称(规范)
prcp_df['prcp_0.1deg'] = np.nan

# 将0.1deg栅格中对应站点的值提取
for date_step in range(len(da)):
    cur_img = da[date_step, :, :]
    cur_date = Config.start_date + relativedelta(months=date_step)
    cur_prcp_df = prcp_df[prcp_df['date'] == cur_date]
    for cur_ix, cur_row in cur_prcp_df.iterrows():
        cur_lat, cur_lon = cur_row.Lat, cur_row.Lon
        prcp_df.loc[cur_ix, 'prcp_0.1deg'] = cur_img.sel(lat=cur_lat, lon=cur_lon, method='nearest', tolerance=0.1)
    print('站点与栅格匹配: {}'.format(cur_date))
out_path = os.path.join(out_dir, 'all_prcp_st.xlsx')
prcp_df['YM'] = prcp_df['date'].dt.strftime('%Y/%m')
prcp_df.to_excel(out_path, index=False)

# 计算每个站点精度指标
st_ids = prcp_df['station_id'].unique()
eval_df = pd.DataFrame(index=st_ids, columns=['r2_st_0.1deg', 'r2_st_1km', 'r2_0.1deg_1km'])
for cur_st_id in st_ids:
    cur_prcp_df = prcp_df[prcp_df['station_id'] == cur_st_id]
    cur_prcp_df = cur_prcp_df.dropna(axis=0, how='any', subset=['prcp_st', 'prcp_0.1deg', 'prcp_1km'])
    if len(cur_prcp_df) <= 3:
        print('跳过: {}'.format(cur_st_id))
        continue
    eval_df.loc[cur_st_id, 'r2_st_0.1deg'] = r2_score(cur_prcp_df['prcp_st'], cur_prcp_df['prcp_0.1deg'])
    eval_df.loc[cur_st_id, 'r2_st_1km'] = r2_score(cur_prcp_df['prcp_st'], cur_prcp_df['prcp_1km'])
    eval_df.loc[cur_st_id, 'r2_0.1deg_1km'] = r2_score(cur_prcp_df['prcp_0.1deg'], cur_prcp_df['prcp_1km'])
out_path = os.path.join(out_dir, 'eval_st.xlsx')
eval_df.to_excel(out_path)

# 计算每个月的精度指标(后续没有明确表明指标就代表计算的是预测降水值与站点实测降水值之间的指标)
date_ids = prcp_df['Month'].unique()
eval_df = pd.DataFrame(index=date_ids, columns=['R2', 'RMSE', 'spearmanr', 'bias1', 'bias2'])
for cur_date_id in date_ids:
    cur_prcp_df = prcp_df[prcp_df['Month'] == cur_date_id]
    cur_prcp_df = cur_prcp_df.dropna(axis=0, how='any', subset=['prcp_st', 'prcp_0.1deg', 'prcp_1km'])
    if len(cur_prcp_df) <= 3:
        print('跳过: {}'.format(cur_date_id))
        continue
    y_true, y_pred = cur_prcp_df['prcp_st'], cur_prcp_df['prcp_1km']
    eval_df.loc[cur_date_id, 'R2'] = r2_score(y_true, y_pred)
    eval_df.loc[cur_date_id, 'RMSE'] = root_mean_squared_error(y_true, y_pred)
    corr, _ = spearmanr(y_true, y_pred)
    eval_df.loc[cur_date_id, 'spearmanr'] = corr
    eval_df.loc[cur_date_id, 'bias1'] = np.sum(y_true) / np.sum(y_pred) - 1
    eval_df.loc[cur_date_id, 'bias2'] = np.sum(y_true - y_pred) / len(y_true)
    cur_prcp_df.to_excel(os.path.join(out_dir, 'temp.xlsx'), index=False)


# 计算2019-2023年西南地区各个年份下各个季节的平均降水量
prcp_season_mean_box = pd.DataFrame(index=np.arange(2019, 2024), columns=['MAM', 'JJA', 'SON', 'DJF'])
prcp_1km_pred = xr.open_dataarray(prcp_1km_pred_path)
years, months = prcp_1km_pred.date.dt.year.values, prcp_1km_pred.date.dt.month.values
coord_year_season = xr.DataArray(
    data=[extract_year_season(y, m) for y, m in zip(years, months)],
    dims=['date'],
    coords={'date': prcp_1km_pred.date}
)
prcp_1km_pred = prcp_1km_pred.assign_coords(year_season=coord_year_season)
prcp_season_mean = prcp_1km_pred.groupby('year_season').sum(dim='date')
prcp_season_mean = prcp_season_mean.mean(dim=['lat', 'lon'])
for cur_item in prcp_season_mean:
    year, season = cur_item.year_season. values.item().split('.')
    if int(year) not in prcp_season_mean_box.index:
        continue
    prcp_season_mean_box.loc[int(year), season] = cur_item.values.item()
out_path = os.path.join(out_dir, 'prcp_season_mean_yearly.xlsx')
prcp_season_mean_box.to_excel(out_path)

# 掩膜