# @Author  : ChaoQiezi
# @Time    : 2025/3/28 上午10:44
# @Email   : chaoqiezi.one@qq.com
# @FileName: Config

"""
This script is used to 存储配置文件
"""

import os
from pathlib import Path
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
mpl.use('QtAgg')  # 防止绘制图表报错
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

# 设置项目根文件夹为当前工作目录
root_dir = Path(__file__).resolve().parent
os.chdir(root_dir)

# IDM路径
idm_path = r"D:\Softwares\IDM\Internet Download Manager\IDMan.exe"
region_path = r'E:\SoftwaresStorage\ArcGISPro_Storage\Projects\PrecipitationDownscaling\research_region\research_region_block.shp'
model_dir = os.path.join(root_dir, 'Asset', 'models')
# 资源文件夹
resources_dir = os.path.join(root_dir, 'Resources')

# 数据下载-时间范围
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
# ERA5的api和key
my_url = "https://cds.climate.copernicus.eu/api"  # api链接
my_key = "c70a112c-b210-492d-a7a0-29a7c5356820"  # API密钥(Mine)
# 下载参数
concurrent_downloads = 6  # 并行下载文件个数
moniter_interval = 1  # 监测时间间隔, 每隔1s监测一次
lon_lat_extent = [34.5, 97.3, 21.1, 110.5]  # (最北, 最西, 最南, 最东), 下载范围
# 下载进度条样式
bar_format = "{desc}: {percentage:.0f}%|{bar}| [{n_fmt}/{total_fmt}] [已用时间:{elapsed}, 剩余时间:{remaining}, {postfix}]"
# 初始化ERA5的下载请求
request = {
    "variable": [],
    "year": '',
    "month": '',
    "day": [],
    "time": [  # 每小时
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "area": [90, -180, -90, 180],  # 默认全球
    "data_format": "netcdf",  # 输出格式为NetCDF4格式(.nc文件), 可选('grib', 'netcdf')
    "download_format": "unarchived"  # 不压缩下载, 可选('unarchived', 'zip')
}

# 模型
random_state = 42  # 随机种子
class_name = 'prcp_class'
