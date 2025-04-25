# @Author  : ChaoQiezi
# @Time    : 2025/4/1 上午12:22
# @Email   : chaoqiezi.one@qq.com
# @FileName: mod13a3_download

"""
This script is used to 批量下载MOD05-水汽数据

MODIS/Terra Total Precipitable Water Vapor 5-Min L2 Swath 1km and 5km
"""

from Src.utils import DownloadManager

# 准备
links_path = r"E:\ChaoQiezi\下载\3486773548-download.txt"
out_dir = r"G:\MOD05\MOD05_IDM"
# 下载
downloader = DownloadManager(out_dir, links_path=links_path, concurrent_downloads=12, monitor_interval=10)
downloader.download()
