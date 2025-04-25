# @Author  : ChaoQiezi
# @Time    : 2025/4/21 下午8:09
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: era5_download

"""
This script is used to 下载ERA5风向u和v
"""

import os
import cdsapi
from datetime import datetime
from dateutil.relativedelta import relativedelta

from Src.utils import DownloadManager, generate_request
import Config

# 准备
out_dir = r'G:\ERA5'  # 输出nc文件的路径(自行修改)
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
var_names = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
# area_extent = [34.5, 97.3, 21.1, 110.5]  # (最北, 最西, 最南, 最东), 见Config.lon_lat_extent
dataset_name = "reanalysis-era5-land"  # era5-land再分析数据集名称

# 下载
c = cdsapi.Client(url=Config.my_url, key=Config.my_key)
rd = relativedelta(end_date, start_date)
month_range = rd.years * 12 + rd.months + 1
for var_name in var_names:
    cur_out_dir = os.path.join(out_dir, var_name)
    status_path = os.path.join(Config.resources_dir, '{}_download_status.json'.format(var_name))  # 存储下载链接和状态的文件(下载过程中勿删除)
    downloader = DownloadManager(cur_out_dir, status_path=status_path)
    for cur_month in range(month_range):
        cur_date = start_date + relativedelta(months=cur_month)
        cur_filename = '{}_{}{:02}.nc'.format(var_name, cur_date.year, cur_date.month)
        add_bool, cur_item = downloader.is_add_link(filename=cur_filename)
        if not add_bool:
            cur_request = generate_request(var_name, cur_date, Config.lon_lat_extent)
            cur_url = c.retrieve(dataset_name, cur_request).location
            downloader.add_link(cur_url, cur_filename)

        if cur_month % 6 == 0:
            downloader.download()



