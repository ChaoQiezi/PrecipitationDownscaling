# @Author  : ChaoQiezi
# @Time    : 2025/4/23 上午3:59
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: station_preprocessing

"""
This script is used to 气象站点数据预处理: 经纬度匹配, 插值
"""

import re
import os
import pandas as pd
from glob import glob

import Config


# 准备
station_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Station\SURF_CHN_MUL_HOR_STATION.xlsx"
prcp_path = r"E:\Datasets\Objects\PrecipitationDownscaling\Station\PRCPsc164_monthly_sum.csv"

# 匹配经纬度
st = pd.read_excel(station_path, dtype={'区站号': str})
st = st.dropna(axis=0)  # 去除无效的行
prcp = pd.read_csv(prcp_path)
prcp = prcp.melt(['Year', 'Month', 'YM'], var_name='station_id', value_name='prcp')  # 宽格式转化为长格式
prcp['station_id'] = prcp['station_id'].str[1:]
a1 = pd.merge(prcp, st, 'left', left_on='station_id', right_on='区站号')



