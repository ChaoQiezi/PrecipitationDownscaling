# @Author  : ChaoQiezi
# @Time    : 2025/4/23 上午3:59
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: station_preprocessing

"""
This script is used to 气象站点数据
"""

import re
import os
from glob import glob

import Config

# 准备
in_dir = r'G:\NOAA\2019'
csv_paths = glob(os.path.join(in_dir, '*.csv'))

valid_paths = []
for csv_path in csv_paths:
    cur_filename = os.path.basename(csv_path)
    match = re.search(r"\((-?\d+\.\d+),(-?\d+\.\d+)\)", '54423099999_CHENGDE_CHINA_(117.9166666,40.9666666).csv')
    lon, lat = float(match.group(1)), float(match.group(2))
    lat_max, lon_min, lat_min, lon_max = Config.lon_lat_extent
    if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
        valid_paths.append(csv_path)


