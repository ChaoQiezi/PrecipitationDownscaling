# @Author  : ChaoQiezi
# @Time    : 2025/4/28 下午1:27
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: rf_train

"""
This script is used to 训练随机森林模型
"""


import os
import joblib  # 保存模型
import numpy as np
import xarray as xr
from rasterio.plot import show
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score  # 分割数据集, 参数寻优, 交叉验证
from sklearn.ensemble import RandomForestRegressor  # 随机回归模型
from scipy.stats import randint, uniform
from sklearn.metrics import r2_score, make_scorer, mean_squared_error

import Config


# 准备
train_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Samples\rf_train_samples.nc"
force_train = True

# 模型输入预准备
da = xr.open_dataarray(train_path)
cur_da = da.drop_sel(var='prcp_class')  # 去除prcp_class分级
# 整理shape
cur_da = cur_da.stack(sample=['date', 'lat', 'lon']).transpose('sample', 'var')  # shape=(date, var, lat, lon) ==> shape=(样本数, var)
cur_da = cur_da.dropna(dim='sample', how='any')  # 去除存在无效值的样本
# 创建XY
x = cur_da.sel(var=cur_da.coords['var'] != 'prcp').values
y = cur_da.sel(var=cur_da.coords['var'] == 'prcp').values.ravel()  # ravel去除冗余维度
# 数据集分割(训练:验证=8:2)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=Config.random_state)

# 模型训练
cur_model_path = os.path.join(Config.model_dir, 'rf.pkl')
if os.path.exists(cur_model_path) and not force_train:
    rf = joblib.load(cur_model_path)
else:
    rf = RandomForestRegressor(  # 定义模型
        n_estimators=500,  # 初始树的数量
        bootstrap=True,  # 启用 Bootstrap 采样
        oob_score=True,  # 启用 OOB 评估
        n_jobs=-1,  # 使用所有 CPU 核心
        random_state=Config.random_state  # 固定随机种子
    )
    rf.fit(x_train, y_train)
    joblib.dump(rf, cur_model_path)

# 评估
print(r2_score(y_test, rf.predict(x_test)))
# print(r2_score(y_test, best_rf.predict(x_test)))


# # 将预测值恢复原形状
# samples = cur_da['sample'].to_index()
# dates = samples.get_level_values('date')
# lats = samples.get_level_values('lat'
# lons = samples.get_level_values('lon')
# y_pred_reconstructed = xr.full_like(da.isel(var=0).drop_vars('var'), np.nan)
# date_ix = y_pred_reconstructed.date.to_index().get_indexer(dates)
# lat_ix = y_pred_reconstructed.lat.to_index().get_indexer(lats)
# lon_ix = y_pred_reconstructed.lon.to_index().get_indexer(lons)
# y_pred = rf.predict(x)
# y_pred_reconstructed.values[date_ix, lat_ix, lon_ix] = y_pred

print('训练完成.')