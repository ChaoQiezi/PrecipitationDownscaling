# @Author  : ChaoQiezi
# @Time    : 2025/4/1 上午12:22
# @Email   : chaoqiezi.one@qq.com
# @FileName: mod13a3_download

"""
This script is used to 批量下载MOD13A3-植被指数(NDVI/EVI)数据

MODIS/Terra Vegetation Indices Monthly L3 Global 1km SIN Grid V061
"""

from Src.utils import DownloadManager

# 准备
links_path = r'F:\PyProJect\PrecipitationDownscaling\Resources\urls\MOD13A3_ndvi_evi.txt'
out_dir = r'G:\MOD13A3'
# 下载
downloader = DownloadManager(out_dir, links_path=links_path, concurrent_downloads=6, monitor_interval=30)
downloader.download()