# @Author  : ChaoQiezi
# @Time    : 2025/3/31 下午11:37
# @Email   : chaoqiezi.one@qq.com
# @FileName: gpm_download

"""
This script is used to 批量下载GPM降水数据

GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07 (GPM_3IMERGM) at GES DISC
"""

from Src.utils import DownloadManager

# 准备
links_path = r'F:\PyProJect\PrecipitationDownscaling\Resources\urls\GPM_precipitation.txt'
out_dir = r'G:\GPM_IMERG_V07\Final_Month'
# 下载
downloader = DownloadManager(out_dir, links_path=links_path, concurrent_downloads=6, monitor_interval=30)
downloader.download()
