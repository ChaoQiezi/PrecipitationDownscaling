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
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
from pathlib import Path

import Config
from Src.utils import extract_year_season, extract1d_lons_lats, mask_metric

# 准备
train_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_train_samples.nc"
prcp_1km_pred_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf_test_pred_by_residual.nc"
prcp_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\table\pred_prcp_st.xlsx"
out_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\Result\table"

# 读取和预处理
prcp_df = pd.read_excel(prcp_path)  # 读取包含各个年份1km和站点
prcp_df = prcp_df[(prcp_df['date'] >= Config.start_date) & (prcp_df['date'] <= Config.end_date)]  #提取指定时间范围的df
prcp_df.rename(columns={'prcp_pred': 'prcp_1km', 'prcp': 'prcp_st'}, inplace=True)  # 更换列名称(规范)
prcp_df['prcp_0.1deg'] = np.nan

# 将0.1deg栅格中对应站点的值提取
da = xr.open_dataarray(train_path).sel(var='prcp').drop_vars('var')  # 读取0.1deg的训练集的降水数据
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

# 01 计算每个站点精度指标
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

# 02 计算每个月的精度指标(后续没有明确表明指标就代表计算的是预测降水值与站点实测降水值之间的指标)
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


# 03 计算2019-2023年西南地区年平均降水量
# 整区域计算
"""
由于绘姐给定的气象站点只有19-20年,因此这里得到年平均降水量,只能获取得到1km和0.1deg的,且不能使用上述的prcp_df(因为
其只包含19-20年数据)
"""
prcp_year_1km_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf\yearly\prcp_year_mean.nc"
prcp_year_0p1deg_path = r"E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG\0.1deg\yearly\prcp_year_mean.nc"
prcp_year_1km = xr.open_dataarray(prcp_year_1km_path)
prcp_year_0p1deg = xr.open_dataarray(prcp_year_0p1deg_path)
prcp_year_1km = prcp_year_1km.mean(dim=['lat', 'lon'])
prcp_year_0p1deg = prcp_year_0p1deg.mean(dim=['lat', 'lon'])
prcp_year_df = pd.DataFrame(index=np.arange(2019, 2024), columns=['prcp_0.1deg', 'prcp_1km'])
prcp_year_df['prcp_1km'] = prcp_year_1km.values
prcp_year_df['prcp_0.1deg'] = prcp_year_0p1deg.values
out_path = os.path.join(out_dir, 'prcp_year_mean.xlsx')
prcp_year_df.to_excel(out_path)
corr, p = pearsonr(prcp_year_df['prcp_0.1deg'], prcp_year_df['prcp_1km'])
# 分地区计算
region_names = ['Sichuan', 'Chongqing', 'Yunnan', 'Guizhou']
out_path = os.path.join(out_dir, 'prcp_year_mean_region.xlsx')
with pd.ExcelWriter(out_path, mode='w') as writer:
    for region_name in region_names:
        cur_prcp_year_1km_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf\yearly\prcp_year_mean_{}.nc".format(region_name)
        cur_prcp_year_0p1deg_path = r"E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG\0.1deg\yearly\prcp_year_mean_{}.nc".format(region_name)
        prcp_year_1km = xr.open_dataarray(cur_prcp_year_1km_path)
        prcp_year_0p1deg = xr.open_dataarray(cur_prcp_year_0p1deg_path)
        prcp_year_1km = prcp_year_1km.mean(dim=['lat', 'lon'])
        prcp_year_0p1deg = prcp_year_0p1deg.mean(dim=['lat', 'lon'])
        prcp_year_df = pd.DataFrame(index=np.arange(2019, 2024), columns=['prcp_0.1deg', 'prcp_1km'])
        prcp_year_df['prcp_1km'] = prcp_year_1km.values
        prcp_year_df['prcp_0.1deg'] = prcp_year_0p1deg.values
        prcp_year_df.to_excel(writer, sheet_name=region_name)
        corr, p = pearsonr(prcp_year_df['prcp_0.1deg'], prcp_year_df['prcp_1km'])
        print('{}-R(0.1deg, 1km) = {:0.2}, p={:0.2}'.format(region_name, corr, p))


# 04 计算2019-2023年西南地区各个年份下各个季节的平均降水量
# 整体计算
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
# 分区域计算
# 分区域计算-掩膜各个城市
mask_dir = r'E:\SoftwaresStorage\ArcGISPro_Storage\Projects\PrecipitationDownscaling\research_region'
template_1km_path = r"E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif"  # 作为写入tiff文件的参考地理文件使用
cur_out_dir = Path(prcp_1km_pred_path).parent
temp_path = r'E:\MyTEMP\temp.tif'
prcp_1km = xr.open_dataarray(prcp_1km_pred_path)
region_names = ['Sichuan', 'Chongqing', 'Yunnan', 'Guizhou']
for region_name in region_names:
    masked_metric_list = []
    filename = region_name + '.shp'
    mask_path = os.path.join(mask_dir, filename)
    for date_ix in prcp_1km.date.values:
        cur_metric = prcp_1km.loc[date_ix, :, :].values
        masked_img = mask_metric(cur_metric, template_1km_path, mask_path, out_path=temp_path)
        masked_metric_list.append(masked_img.GetRasterBand(1).ReadAsArray())
        masked_img = None
    lons, lats = extract1d_lons_lats(temp_path)
    da = xr.DataArray(
            data=masked_metric_list,
            dims=['date', 'lat', 'lon'],
            coords={
                'date': prcp_1km.date.values,
                'lat': lats[::-1],  # 数组从上往下第一行区域是纬度最高的地方, 所以纬度要从大到小逆序传入!
                'lon': lons,
            },
            name='date_{}'.format(region_name)
        )
    out_path = os.path.join(cur_out_dir, 'rf_test_pred_by_residual_{}.nc'.format(region_name))
    da.to_netcdf(out_path)
    print('掩膜处理(date): {}'.format(region_name))
# 分区域计算-分省市计算2019-2023年西南地区各个年份下各个季节的平均降水量
region_names = ['Sichuan', 'Chongqing', 'Yunnan', 'Guizhou']
out_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Result\table\prcp_season_mean_yearly_region.xlsx"
with pd.ExcelWriter(out_path, mode='w') as writer:
    for region_name in region_names:
        prcp_season_mean_box = pd.DataFrame(index=np.arange(2019, 2024), columns=['MAM', 'JJA', 'SON', 'DJF'])
        cur_prcp_1km_pred_path = os.path.join(out_dir, 'rf_test_pred_by_residual_{}.nc'.format(region_name))
        prcp_1km_pred = xr.open_dataarray(cur_prcp_1km_pred_path)
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
            year, season = cur_item.year_season.values.item().split('.')
            if int(year) not in prcp_season_mean_box.index:
                continue
            prcp_season_mean_box.loc[int(year), season] = cur_item.values.item()
        prcp_season_mean_box.to_excel(writer, sheet_name=region_name)
        print('处理: {}'.format(region_name))


# 05 计算2019-2023年西南地区各个省市下各个季节的平均降水量
region_names = ['Sichuan', 'Chongqing', 'Yunnan', 'Guizhou']
prcp_season_mean_box = pd.DataFrame(index=['MAM', 'JJA', 'SON', 'DJF'], columns=['西南', *region_names])
prcp_season_mean_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf\seasonally\prcp_season_mean.nc"
prcp_season_mean = xr.open_dataarray(prcp_season_mean_path)
prcp_season_mean_box.loc[prcp_season_mean.season.values, '西南'] = prcp_season_mean.mean(dim=['lat', 'lon']).values
for region_name in region_names:
    cur_prcp_season_mean_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf\seasonally\prcp_season_mean_{}.nc".format(region_name)
    prcp_season_mean = xr.open_dataarray(cur_prcp_season_mean_path)
    prcp_season_mean_box.loc[prcp_season_mean.season.values, region_name] = prcp_season_mean.mean(dim=['lat', 'lon'])
prcp_season_mean_box.index = ['春季', '夏季', '秋季', '冬季']
prcp_season_mean_box.columns = ['西南', '四川省', '重庆市', '云南省', '贵州省']
out_path = os.path.join(out_dir, 'prcp_season_mean_region.xlsx')
prcp_season_mean_box.to_excel(out_path)


# 06 (分地区)计算2019-2023年西南地区多年月均降水量
region_names = ['Sichuan', 'Chongqing', 'Yunnan', 'Guizhou']
prcp_month_mean_df = pd.DataFrame(index=np.arange(1, 13), columns=['西南地区', *region_names])
# 6.1 西南地区
prcp_month_mean_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf\monthly\prcp_month_mean.nc"
prcp_month_mean_da = xr.open_dataarray(prcp_month_mean_path)
prcp_month_mean_df['西南地区'] = prcp_month_mean_da.mean(dim=['lat', 'lon']).values
# 6.2 分地区
in_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\Prediction\rf\monthly'
for region_name in region_names:
    cur_nc_path = os.path.join(in_dir, 'prcp_month_mean_{}.nc'.format(region_name))
    cur_da = xr.open_dataarray(cur_nc_path)
    cur_da = cur_da.mean(dim=['lat', 'lon'])
    prcp_month_mean_df[region_name] = cur_da.values
out_path = os.path.join(out_dir, 'prcp_month_mean.xlsx')
prcp_month_mean_df.to_excel(out_path)
