# @Author  : ChaoQiezi
# @Time    : 2025/4/28 下午1:27
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: rf_train

"""
This script is used to 
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

# 训练
da = xr.open_dataarray(train_path)

# 去除prcp_class分级
cur_da = da.drop_sel(var='prcp_class')
# 整理shape
cur_da = cur_da.stack(sample=['date', 'lat', 'lon']).transpose('sample', 'var')
cur_da = cur_da.dropna(dim='sample', how='any')
# 创建XY
x = cur_da.sel(var=cur_da.coords['var'] != 'prcp').values
y = cur_da.sel(var=cur_da.coords['var'] == 'prcp').values.ravel()  # ravel去除冗余维度
# 数据集分割
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=Config.random_state)

# 模型训练
cur_model_cv_path = os.path.join(Config.model_dir, 'rf_cv.pkl')
cur_model_path = os.path.join(Config.model_dir, 'rf.pkl')
# if os.path.exists(cur_model_cv_path):  # 若模型已训练则跳过
#     best_rf = joblib.load(cur_model_cv_path)
# else:
#     rf = RandomForestRegressor(  # 定义模型
#         n_estimators=200,  # 初始树的数量
#         bootstrap=True,  # 启用 Bootstrap 采样
#         oob_score=True,  # 启用 OOB 评估
#         n_jobs=-1,  # 使用所有 CPU 核心
#         random_state=Config.random_state  # 固定随机种子
#     )
#     param_dist = {  # 搜寻的参数空间
#         "n_estimators": [300, 400, 500],  # 树的数量
#         "max_depth": [None] + np.arange(10, 30, 5).tolist(),  # 树的最大深度
#         "max_features": [0.6, 0.7, 0.8, 1],  # 特征采样策略
#         "min_samples_split": randint(2, 20),  # 节点分裂最小样本数
#         "min_samples_leaf": randint(1, 10),  # 叶子节点最小样本数
#     }
#     # 超参数寻优
#     search = RandomizedSearchCV(
#         rf, param_dist,
#         n_iter=50,  # 随机迭代次数, 从param_dist中随机选取50个组合
#         cv=5,  # 交叉验证的折数(将训练集分成5份, 4份训练1份验证, 五份每一份都可以作为验证因此可以不重复训练五次)
#         # scoring='neg_mean_squared_error',  # 回归任务指标,要求是得分越高越好, 因此取MSE的负数
#         scoring='r2',
#         verbose=2,  # 日志输出的详细度(0: 不输出<静默>, 1: 显示进度条, 2: 详细显示)
#         random_state=Config.random_state,  # 随机种子
#         n_jobs=-1  # 使用所有CPU
#     )
#     search.fit(x_train, y_train)  # 训练
#     best_rf = search.best_estimator_
#
#     # 保存模型
#     joblib.dump(best_rf, cur_model_cv_path)

if os.path.exists(cur_model_path):
    rf = joblib.load(cur_model_path)
else:
    rf = RandomForestRegressor(  # 定义模型
        n_estimators=200,  # 初始树的数量
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
# lats = samples.get_level_values('lat')
# lons = samples.get_level_values('lon')
# y_pred_reconstructed = xr.full_like(da.isel(var=0).drop_vars('var'), np.nan)
# date_ix = y_pred_reconstructed.date.to_index().get_indexer(dates)
# lat_ix = y_pred_reconstructed.lat.to_index().get_indexer(lats)
# lon_ix = y_pred_reconstructed.lon.to_index().get_indexer(lons)
# y_pred = rf.predict(x)
# y_pred_reconstructed.values[date_ix, lat_ix, lon_ix] = y_pred

print('训练完成.')