# @Author  : ChaoQiezi
# @Time    : 2025/4/1 上午12:22
# @Email   : chaoqiezi.one@qq.com
# @FileName: mod11a2_download

"""
This script is used to 批量下载MOD11A2-地表温度数据

MODIS/Terra Land Surface Temperature/Emissivity 8-Day L3 Global 1km SIN Grid V061
"""

from Src.utils import DownloadManager

# 准备
links_path = r'F:\PyProJect\PrecipitationDownscaling\Resources\urls\MOD11A2_temperature.txt'
out_dir = r'G:\MOD11A2'
# 下载
downloader = DownloadManager(out_dir, links_path=links_path, concurrent_downloads=6, monitor_interval=30)
downloader.download()