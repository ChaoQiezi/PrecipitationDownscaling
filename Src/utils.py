# @Author  : ChaoQiezi
# @Time    : 2025/3/31 下午11:21
# @Email   : chaoqiezi.one@qq.com
# @FileName: utils

"""
This script is used to 
"""

import os
import time
import re
import json
from typing import Union
from numpy import ceil, floor
from tqdm import tqdm
from urllib.parse import urlparse  # 解析下载链接获取下载文件的状态(例如默认文件名)
from subprocess import call
from osgeo import gdal, osr
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.plot import show
import h5py
import netCDF4 as nc
import numpy as np
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pyproj import Transformer
from math import ceil
from pyhdf.SD import SD
from matplotlib import pyplot as plt
import calendar
from datetime import datetime

import Config

gdal.UseExceptions()  # 使用异常错误机制(存在错误即报错而不是继续往下执行代码)


def _get_filename(url: str):
    """
    基于下载链接获取默认的文件名称
    :param url: 下载链接
    :return: None
    """
    return os.path.basename(urlparse(url).path)


def generate_request(var_name, datetime, area=None, request=Config.request):
    """
    生成ERA5的请求
    :param var_name: 下载的变量名称
    :param datetime: 下载的时间范围(依据datetime获取年月,下载该月份所有数据)
    :param area:
    :param request:
    :return:
    """

    # 下载请求
    request['year'] = '{}'.format(datetime.year)
    request['month'] = '{:02}'.format(datetime.month)
    days_of_month = calendar.monthrange(datetime.year, datetime.month)[1]  # 当前月份的天数
    request['day'] = ['{:02}'.format(_day) for _day in range(1, days_of_month + 1)]
    request['variable'] = [var_name]

    # 地理范围(默认全球)
    if area is not None:
        request['area'] = area

    return request


class DownloadManager:
    def __init__(self, out_dir, links_path=None, status_path=None, concurrent_downloads=Config.concurrent_downloads,
                 monitor_interval=Config.moniter_interval):
        """
        初始化类
        :param out_dir: 下载文件的输出目录
        :param links_path: 存储下载链接的txt文件(一行一个下载链接)
        :param status_path: 存储结构化下载链接的json文件(用于存储下载链接和状态的json文件)
        """

        # 存储下载状态的json文件
        if status_path is None:
            status_path = os.path.join(Config.Resources_dir, 'links_status.json')
        self.status_path = status_path
        # 下载文件的输出路径
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        # 下载状态
        self.downloading_links = list()
        self.pending_links = list()
        self.completed_links = list()
        self.links = list()
        self.pbar = None  # 下载进度条, 执行self.download()时触发
        # 下载参数
        self.concurrent_downloads = concurrent_downloads  # 同时下载文件数量(并发量)
        self.monitor_interval = monitor_interval  # 监测下载事件的时间间隔, 单位:秒/s
        self.downloaded_count = len(self.completed_links)  # 已下载数
        self.remaining_downloads = len(self.links) - self.downloaded_count  # 未下载数
        self.link_count = len(self.links)

        # 初始化下载状态
        if links_path is not None:  # 将存储下载链接的txt文件存储为结构化json文件
            self._init_save(links_path)
        elif os.path.exists(self.status_path):
            with open(self.status_path, 'r') as f:
                links_status = json.load(f)
                self.downloading_links = links_status['downloading_links']
                self.pending_links = links_status['pending_links']
                self.completed_links = links_status['completed_links']
                self.links = links_status['links']
                self._update()
        else:
            self._update()

    def _init_save(self, links_path):
        """
        从存储下载链接的txt文件中初始化下载链接及其下载状态等参数
        :param links_path: 存储下载链接的txt文件
        :return: None
        """

        with open(links_path, 'r') as f:
            urls = []
            for line in f:
                if not line.startswith('http') and not line.startswith('ftp'):
                    continue
                urls.append({
                    'url': line.rstrip('\n'),
                    'filename': self._get_filename(line.rstrip('\n'))
                })

        self.links = urls.copy()
        self.pending_links = urls.copy()
        """
        # 必须使用copy(), 否则后续对self.pending_links中元素操作, 会影响self.links的元素, 因为二者本质上都是指向(id相同)同一个列表urls
        self.links = urls
        self.pending_links = urls
        """

        self._update()

    def _update(self, downloading_links=None, pending_links=None, completed_links=None, links=None):
        """更新下载链接的状态位置并保存"""

        if downloading_links is None:
            downloading_links = self.downloading_links
        if pending_links is None:
            pending_links = self.pending_links
        if completed_links is None:
            completed_links = self.completed_links
        if links is None:
            links = self.links

        self.downloaded_count = len(self.completed_links)
        self.remaining_downloads = len(self.links) - self.downloaded_count
        self.link_count = len(self.links)

        with open(self.status_path, 'w') as f:
            json.dump({
                'downloading_links': downloading_links,
                'pending_links': pending_links,
                'completed_links': completed_links,
                'links': links
            }, f, indent=4)  # indent=4表示缩进为4,让排版更美观

    def add_link(self, link: str, filename=None):
        """
        添加新链接
        :param link: 需要添加的一个链接
        :param filename: 该链接对应下载文件的输出文件名
        :return: None
        """

        # 结构化下载链接
        new_item = self._generate_item(link, filename)

        # 添加下载链接到links
        if new_item not in self.links:
            self.links.append(new_item)
            self.pending_links.append(new_item)

        self._update()

    def _get_filename(self, url):
        """获取下载链接url对应的默认文件名称"""

        return os.path.basename(urlparse(url).path)

    def _generate_item(self, link: str, filename=None):
        """基于下载链接生成item"""

        item = {
            'url': link,
        }
        if filename is not None:
            item['filename'] = filename
        else:
            item['filename'] = self._get_filename(link)

        return item

    def _init_download(self):
        """
        初始化下载链接的状态并启动下载
        :return:
        """

        # self.links复制一份到pending_links中
        self.pending_links = self.links.copy()

        self._pending2downloading()  # 将<等待下载队列>中的链接添加到<正在下载队列>去
        # call([Config.idm_path, '/s'])  # 启动IDM中<主要下载队列>的所有待下载链接

    def download(self):
        """
        对此前加入的所有url进行下载
        :return:
        """

        try:
            self.pbar = tqdm(total=self.link_count, desc='下载', bar_format=Config.bar_format, colour='blue')
            self._init_download()
            self._monitor()
        except KeyboardInterrupt:
            print('您已中断下载程序; 下次下载将继续从({}/{})处下载...'.format(self.downloaded_count, self.link_count))
            exit(1)  # 错误退出
        except Exception as e:
            print(
                '下载异常错误: {};\n下次下载将继续从({}/{})处下载...'.format(e, self.downloaded_count, self.link_count))
            exit(1)  # 错误退出
        finally:
            self._update()  # 无论是否发生异常, 最后都必须保存当前下载状态, 以备下次下载继续从断开处进行

    def download_single(self, url, filename=None, wait_time=None):
        """
        对输入的单个url进行下载, 最好不要与download()方法连用
        :param url: 所需下载的文件链接
        :param filename: 输出的文件名称
        :return:
        """

        if filename is None:
            filename = self._get_filename(url)

        # 判断当前url文件是否已经下载
        out_path = os.path.join(self.out_dir, filename)
        if os.path.exists(out_path):
            if wait_time is not None:
                return wait_time

        call([Config.idm_path, '/d', url, '/p', self.out_dir, '/f', filename, '/a', '/n'])
        call([Config.idm_path, '/s'])

        if wait_time is not None:
            return wait_time + 0
        """
        IDM命令行说明:
        cmd: idman /s
        /s: 开始(start)下载添加IDM中下载队列中的所有文件
        cmd: idman /d URL [/p 本地_路径] [/f 本地_文件_名] [/q] [/h] [/n] [/a]
        /d URL: 从下载链接url中下载文件
        /p 本地_路径: 下载好的文件保存在哪个本地路径(文件夹路径/目录)
        /f 本地_文件_名: 下载好的文件输出/保存的文件名称
        /q: IDM 将在成功下载之后退出。这个参数只为第一个副本工作
        /h: IDM 将在正常下载之后挂起您的连接(下载窗口最小化/隐藏到系统托盘)
        /n: IDM不要询问任何问题不要弹窗,安静地/后台地下载
        /a: 添加一个指定的文件, 用/d到下载队列, 但是不要开始下载.(即添加一个下载链接到IDM的下载队列中, 可通过/s启动队列的所有下载链接文件的下载)
        """

    def _monitor(self):
        while True:
            for item in self.downloading_links.copy():  # .copy()是为了防止在循环过程中一边迭代downloading_links一边删除其中元素
                self._check_update_download(item)
            self._update()  # 更新和保存下载状态
            self.pbar.refresh()  # 更新下载进度条状态
            call([Config.idm_path, '/s'])  # 防止IDM意外停止下载

            # 直到等待下载链接和正在下载链接中均无下载链接说明下载完毕.
            if not self.pending_links and not self.downloading_links:
                self.pbar.close()  # 关闭下载进度条
                break

            time.sleep(self.monitor_interval)

    def _check_update_download(self, downloading_item):
        """
        检查当前项是否已经下载, 成功下载则更新该项的状态并返回True, 否则不操作并返回False
        :param downloading_item: <正在下载链接>中的当前项
        :return: Bool
        """

        out_path = os.path.join(self.out_dir, downloading_item['filename'])

        # 检查当前文件是否存在(是否下载)
        if os.path.exists(out_path):  # 存在(即已经下载过了)
            # 更新当前文件的下载状态
            self.completed_links.append(downloading_item)
            self.downloading_links.remove(downloading_item)
            self._update_pbar(downloading_item['filename'])  # 更新下载进度条
            # print('文件: {} - 下载完成({}/{})'.format(downloading_item['filename'], len(self.completed_links), len(self.links)))
            # 从<阻塞/等待下载链接>中取链接到<正在下载链接>中(如果pending_links中还有链接)
            if self.pending_links:
                self._pending2downloading()  # 取<阻塞/等待下载链接>中的链接添加到<正在下载链接>中

            return True
        return False

    def _download(self, item):
        self.download_single(item['url'], item['filename'])

    def _pending2downloading(self):
        """
        从阻塞的<等待下载链接>中取链接<正在下载链接>中,若所取链接已经下载则跳过
        :return:
        """

        for item in self.pending_links.copy():
            out_path = os.path.join(self.out_dir, item['filename'])
            # 判断当前下载链接是否已经被下载
            if os.path.exists(out_path):  # 若当前链接已经下载, 跳过下载并更新其状态
                self.pending_links.remove(item)
                self.completed_links.append(item)
                self._update_pbar(item['filename'])
                continue
            elif self.downloading_links.__len__() < self.concurrent_downloads:  # 若当前链接未被下载且当前下载数量小于并发量
                self.pending_links.remove(item)
                self.downloading_links.append(item)
                self._download(item)
            else:
                # 若elif中不能执行, 说明当前项未下载, 且当前同时下载的文件数量已达到最大, 因此不需要迭代下去了
                break

    def is_add_link(self, item=None, url=None, filename=None):
        """
        依据item/url/filename判断该链接此前已经被添加过, 如果添加过那么返回True, 如果没有被添加过则返回False
        :param item: 形如dict{'url': str, 'filename': str}的item
        :param url: 包含单个下载链接的字符串
        :param filename: 包含输出文件名称的字符串
        :return: (Bool, dict)
        """

        if not self.links:
            return False, {}

        # 依据item判断
        if item is not None:
            for cur_item in self.links:
                if cur_item == item:
                    return True, item
            return False, {}

        # 依据链接判断
        if url is not None:
            for item in self.links:
                if item['url'] == url:
                    return True, item
            return False, {}

        # 依据输出文件名称判断
        if filename is not None:
            for item in self.links:
                if item['filename'] == filename:
                    return True, item
            return False, {}

    def _update_pbar(self, filename):
        """
        更新下载进度条
        :return:
        """

        self.pbar.n = len(self.completed_links)  # 更新已完成地数目
        self.pbar.set_postfix_str('当前下载文件: {}'.format(filename))
        # self.pbar.refresh()  # 立即刷新显示


def hdf2tiff(hdf_path, out_dir, unit_conversion=2):
    """
    将输入的hdf5文件(GPM降水数据)输出为geotiff文件
    :param hdf_path: 输入的hdf5文件的路径
    :param out_dir: 输出的geotiff文件的目录
    :param unit_conversion: 单位换算, 0表示mm/hr, 1表示mm/day(日累计降水量), 2表示mm/月(月累积降水量)
    :return: None
    """

    # 读取hdf文件数据集和基本信息
    with h5py.File(hdf_path, 'r') as f:
        var = f['Grid/precipitation'][0, :, :]  # 获取降水数据集(2D), unit: mm/hr
        lon = f['Grid/lon'][:]  # 获取经度数据集(1D)
        lat = f['Grid/lat'][:]  # 获取纬度数据集(1D)
        time = f['Grid/time'][:][0].item()  # 一个数组中仅有一个时间, 并将该时间转化为py原生整数而非np.int32, 下同

        # 降水数据集翻转
        var = var.T  # 将(col, row) ==> (row, col)
        var = np.flipud(var)  # 将栅格矩阵上下颠倒(原先为上南极下北极)
        # 基本信息
        var_fill_value = f['Grid/precipitation'].attrs['_FillValue'].item()
        init_time = datetime(1980, 1, 6, tzinfo=timezone.utc)  # 创建初始化时间(UTC:格林尼治时间)
        time = init_time + relativedelta(seconds=time)
        row, col = var.shape
        lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
        lon_min, lon_max, lat_min, lat_max = lon_min - 0.05, lon_max + 0.05, lat_min - 0.05, lat_max + 0.05
        """
        下面两个参数通过`LongName`属性获取得到:
        Longitude at the center of\n\t\t\t0.10 degree grid intervals of longitude \n\t\t\tfrom -180 to 180.
        Latitude at the center of\n\t\t\t0.10 degree grid intervals of latitude\n\t\t\tfrom -90 to 90.
        此处的lon和lat中每一个值均表示该值对应像元/栅格网格的中心位置的经纬度值.
        因此如果计算整幅影像的覆盖范围即经纬度范围, 需要加上或者减少半个分辨率(GPM分辨率为0.1°)
        """
        lon_res, lat_res = (lon_max - lon_min) / col, (lat_max - lat_min) / row

        # 无效值设置
        var[var == var_fill_value] = np.nan
        # 计算每月降水量/单位换算 (mm/hr ==> mm/月)
        if unit_conversion == 2:  # 换算为每月降水量
            date_str = os.path.basename(hdf_path).split('3B-MO.MS.MRG.3IMERG.')[1][:6]
            year, month = int(date_str[:4]), int(date_str[4:])
            mdays = calendar.monthrange(year, month)[1]  # 计算当前月份的总天数
            var = var * 24 * mdays  # 计算每月降水量
        elif unit_conversion == 1:  # 换算为每天累计降水量
            var = var * 24
        elif unit_conversion == 0:  # 什么也不需要做
            pass
    # 输出
    out_tiff_name = 'GPM_{}_{:02}.tif'.format(time.year, time.month)
    out_tiff_path = os.path.join(out_dir, out_tiff_name)
    tiff_driver = gdal.GetDriverByName('GTiff')
    ds = tiff_driver.Create(out_tiff_path, col, row, 1,
                            gdal.GDT_Float32)  # (输出路径, x方向大小<即列>, y方向大小<即行>, 波段数, 输出的数据类型<此处浮点型>)
    # 地理参数设置
    geo_transform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]  # 仿射变换参数
    ds.SetGeoTransform(geo_transform)
    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=latlong')  # 规则经纬度坐标系(默认的WGS84坐标系参数)
    ds.SetProjection(srs.ExportToWkt())
    # 写入波段
    b1 = ds.GetRasterBand(1)
    b1.WriteArray(var)  # 写入降水栅格矩阵
    b1.SetNoDataValue(np.nan)  # 设置无效值
    b1.ComputeStatistics(False)  # 计算该波段的统计数据(方便arcmap更好显示), False表示不粗略计算(即精确计算)

    ds.FlushCache()  # 释放内存
    ds = None  # 垃圾回收机制保证完整输出


def nc2tiff(nc_path, tiff_path, var_name):
    """
    将nc格式的ERA文件输出为Geotiff文件
    :param nc_path: 待处理的nc文件路径
    :param tiff_path: Geotiff文件的输出路径
    :return:
    """

    # 获取变量和经纬度
    with nc.Dataset(nc_path) as f:
        var = f.variables[var_name][:]
        lon = f.variables['longitude'][:]
        lat = f.variables['latitude'][:]
    # 获取基本信息
    lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
    lon_res = (lon_max - lon_min) / len(lon)
    lat_res = (lat_max - lat_min) / len(lat)
    time, row, col = var.shape
    # 月均值计算
    var = np.mean(var, axis=0).filled(np.nan)  #并将无效值设置为nan, numpy.ma.MaskedArray ==> numpy.ndarray

    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +R=6367470 +datum=WGS84')
    """见官方文档https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference"""
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(tiff_path, col, row, 1, gdal.GDT_Float32)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform([lon_min, lon_res, 0, lat_max, 0, -lat_res])
    b1 = ds.GetRasterBand(1)
    b1.WriteArray(var)
    b1.SetNoDataValue(np.nan)
    b1.ComputeStatistics(False)  # 计算统计数据,方便显示(False不粗略显示)

    ds = None


def clip_mask(tiff_path, shp_path, out_path, out_res, resample_alg=2, remove_src=False, dst_nodata=None):
    """
    将输入的geotiff文件进行裁剪、掩膜
    :param tiff_path: 输入的geotiff文件的路径
    :param shp_path: 掩膜所需的shp文件的路径
    :param out_path: 处理好的tiff文件的输出路径
    :param out_res: 输出分辨率
    :param resample_alg: 重采样算法(0: 最近邻; 1: 双线性; 2: 三次卷积)
    :param remove_src: 是否删除源文件即tiff_path
    :return: None
    """

    options = gdal.WarpOptions(
        xRes=out_res,
        yRes=out_res,
        resampleAlg=resample_alg,  # 重采样算法
        cutlineDSName=shp_path,
        cropToCutline=True,  # True表示裁剪到矩形范围后继续掩膜, False表示只裁剪到矩形范围
        targetAlignedPixels=True,  # 对齐像元
        dstNodata=dst_nodata,
    )
    ds = gdal.Warp(out_path, tiff_path, options=options)
    ds = None

    if remove_src:
        os.remove(tiff_path)

    with rio.open(out_path) as f:
        b1 = f.read(1)


def img_mosaic(mosaic_paths: list, mosaic_ds_name: str, return_all: bool = True, img_nodata: Union[int, float] = np.nan,
               img_type: Union[np.int32, np.float32, None] = np.float32, unit_conversion: bool = False,
               scale_factor_op: str = 'multiply', mosaic_mode: str = 'last'):
    """
    该函数用于对列表中的所有HDF4文件进行镶嵌
    :param mosaic_paths: 多个HDF4文件路径组成的字符串列表
    :param mosaic_ds_name: 待镶嵌的数据集名称
    :param return_all: 是否一同返回仿射变换、镶嵌数据集的坐标系等参数
    :param img_nodata: 影像中的无效值设置
    :param img_type: 待镶嵌影像的数据类型
    :param unit_conversion: 是否进行单位换算
    :param scale_factor_op: 比例因子的运算符, 默认是乘以(可选: multiply, divide), 该参数尽在unit_conversion为True时生效
    :param mosaic_mode: 镶嵌模式, 默认是Last(即如果有存在像元重叠, mosaic_paths中靠后影像的像元将覆盖其),
        可选: last, mean, max, min, 镶嵌策略默认是last模式,
    :return: 默认返回镶嵌好的数据集
    """

    # 获取镶嵌范围
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for mosaic_path in mosaic_paths:
        hdf = SD(mosaic_path)  # 默认只读
        # 获取元数据
        metadata = hdf.attributes()['StructMetadata.0']
        # 获取角点信息
        ul_pt = [float(x) for x in re.search(r'UpperLeftPointMtrs=\((.*?)\)', metadata).group(1).split(',')]
        lr_pt = [float(x) for x in re.search(r'LowerRightMtrs=\((.*?)\)', metadata).group(1).split(',')]
        x_mins.append(ul_pt[0])
        x_maxs.append(lr_pt[0])
        y_mins.append(lr_pt[1])
        y_maxs.append(ul_pt[1])
    else:
        # 计算分辨率
        col = int(re.search(r'XDim=(.*?)\n', metadata).group(1))
        row = int(re.search(r'YDim=(.*?)\n', metadata).group(1))
        x_res = (lr_pt[0] - ul_pt[0]) / col
        y_res = (ul_pt[1] - lr_pt[1]) / row
        # 如果img_type没有指定, 那么数据类型默认为与输入相同
        if img_type is None:
            img_type = hdf.select(mosaic_ds_name)[:].dtype
        # 获取数据集的坐标系参数并转化为proj4字符串格式
        projection_param = [float(_param) for _param in re.findall(r'ProjParams=\((.*?)\)', metadata)[0].split(',')]
        """
        Sinusoidal Equal Area (INSYS = 16): TPARIN( 1 ) and TPARIN( 6:8 ) used.
            1. Radius of sphere of reference
            2. (unused)
            3. (unused)
            4. (unused)
            5. Longitude of central meridian
            6. Latitude of central meridian
            7. False easting in the same units as the semimajor axis
            8. False northing in the same units as the semimajor axis
            9. (unused)...
        """
        mosaic_img_proj4 = "+proj={} +R={:0.4f} +lon_0={:0.4f} +lat_0={:0.4f} +x_0={:0.4f} " \
                           "+y_0={:0.4f} +units=m +no_defs".format('sinu', projection_param[0], projection_param[4],
                                                                   projection_param[5], projection_param[6],
                                                                   projection_param[7])
        """
        proj4语法:
        +proj=, 坐标系(可填: latlong, sinu等)
        +R=, 参考椭球体的半径<如果椭球体的长半轴和短半轴不一样, 可以分别设置即: +a=, +b=>
        +lon_0=, 中央经线的经度
        +lat_0=, 中央纬线的维度
        +x_0=, 投影坐标系原点的东西偏移(东为正)
        +y_0=, 投影坐标系原点的南北偏移(北为正)
        +units=, 坐标系单位(可填: m, deg等)
        +no_defs, 禁止加载默认参数, 避免冲突
        
        """
        # 关闭文件, 释放资源
        hdf.end()
    x_min, x_max, y_min, y_max = min(x_mins), max(x_maxs), min(y_mins), max(y_maxs)

    # 镶嵌
    col = ceil((x_max - x_min) / x_res)
    row = ceil((y_max - y_min) / y_res)
    mosaic_imgs = []  # 用于存储各个影像
    for ix, mosaic_path in enumerate(mosaic_paths):
        mosaic_img = np.ma.masked_all((row, col), dtype=img_type)
        hdf = SD(mosaic_path)
        target_ds = hdf.select(mosaic_ds_name)
        # 读取数据集和预处理
        target = np.ma.array(target_ds.get())
        valid_range = target_ds.attributes()['valid_range']
        target = np.ma.masked_where((target < valid_range[0]) | (target > valid_range[1]), target)  # 限定有效范围
        if unit_conversion:  # 进行单位换算
            scale_factor = target_ds.attributes()['scale_factor']
            add_offset = target_ds.attributes()['add_offset']
            # 判断比例因子的运算符
            if scale_factor_op == 'multiply':
                target = target * scale_factor + add_offset
            elif scale_factor_op == 'divide':
                target = target / scale_factor + add_offset
            # 计算当前镶嵌范围
        start_row = floor((y_max - (y_maxs[ix] - y_res / 2)) / y_res).astype(int)
        start_col = floor(((x_mins[ix] + x_res / 2) - x_min) / x_res).astype(int)
        end_row = (start_row + target.shape[0]).astype(int)
        end_col = (start_col + target.shape[1]).astype(int)
        mosaic_img[start_row:end_row, start_col:end_col] = target
        mosaic_imgs.append(mosaic_img)

        # 释放资源
        target_ds.endaccess()
        hdf.end()

    # 判断镶嵌模式
    if mosaic_mode == 'last':
        mosaic_imgs = np.ma.stack(mosaic_imgs)
        mosaic_img = mosaic_imgs[0].copy()
        for img in mosaic_imgs:
            mosaic_img[~img.mask] = img[~img.mask]
    elif mosaic_mode == 'mean':
        mosaic_img = np.ma.stack(mosaic_imgs).mean(axis=0).filled(img_nodata)
    elif mosaic_mode == 'max':
        mosaic_img = np.ma.stack(mosaic_imgs).max(axis=0).filled(img_nodata)
    elif mosaic_mode == 'min':
        mosaic_img = np.ma.stack(mosaic_imgs).min(axis=0).filled(img_nodata)
    else:
        raise ValueError('不支持的镶嵌模式: {}'.format(mosaic_mode))

    if return_all:
        extrent_geo = [  # 地理覆盖范围(四个角点, 左上角点顺时针开始)
            [x_min, x_max, x_max, x_min],
            [y_max, y_max, y_min, y_min]
        ]
        return mosaic_img, [x_min, x_res, 0, y_max, 0, -y_res], mosaic_img_proj4, extrent_geo

    return mosaic_img


def img_warp(src_img: np.ndarray, out_path: str, transform: list, src_proj4: str, out_res: Union[float, None] = None,
             nodata_value: Union[int, float] = np.nan, resample: str = 'bilinear', dst_epsg=4326) -> None:
    """
    该函数用于对正弦投影下的栅格矩阵进行重投影(GLT校正), 得到WGS84坐标系下的栅格矩阵并输出为TIFF文件
    :param src_img: 待重投影的栅格矩阵
    :param out_path: 输出路径
    :param transform: 仿射变换参数([x_min, x_res, 0, y_max, 0, -y_res], 旋转参数为0是常规选项)
    :param out_res: 输出的分辨率(栅格方形)
    :param nodata_value: 设置为NoData的数值
    :param out_type: 输出的数据类型
    :param resample: 重采样方法(默认是最近邻, ['nearest', 'bilinear', 'cubic'])
    :param src_proj4: 表达源数据集(src_img)的坐标系参数(以proj4字符串形式)
    :return: None
    """

    # 输出数据类型
    if np.issubdtype(src_img.dtype, np.integer):
        out_type = gdal.GDT_Int32
    elif np.issubdtype(src_img.dtype, np.floating):
        out_type = gdal.GDT_Float32
    else:
        raise ValueError("当前待校正数组类型为不支持的数据类型")
    resamples = {'nearest': gdal.GRA_NearestNeighbour, 'bilinear': gdal.GRA_Bilinear, 'cubic': gdal.GRA_Cubic}
    # 原始数据集创建(正弦投影)
    driver = gdal.GetDriverByName('MEM')  # 在内存中临时创建
    src_ds = driver.Create("", src_img.shape[1], src_img.shape[0], 1, out_type)  # 注意: 先传列数再传行数, 1表示单波段
    srs = osr.SpatialReference()
    srs.ImportFromProj4(src_proj4)
    """
    对于src_proj4, 依据元数据StructMetadata.0知:
        Projection=GCTP_SNSOID; ProjParams=(6371007.181000,0,0,0,0,0,0,0,0,0,0,0,0)
    或数据集属性(MODIS_Grid_8Day_1km_LST/Data_Fields/Projection)知:
        :grid_mapping_name = "sinusoidal";
        :longitude_of_central_meridian = 0.0; // double
        :earth_radius = 6371007.181; // double
    """
    src_ds.SetProjection(srs.ExportToWkt())  # 设置投影信息
    src_ds.SetGeoTransform(transform)  # 设置仿射参数
    b1 = src_ds.GetRasterBand(1)
    b1.WriteArray(src_img)  # 写入数据
    b1.ComputeStatistics(False)  # 计算统计数据, 方便显示
    if nodata_value is not None:
        b1.SetNoDataValue(nodata_value)
    # 重投影信息(默认WGS84)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(dst_epsg)
    # 重投影
    options = gdal.WarpOptions(
        dstSRS=dst_srs,  # 输出的空间参考系
        xRes=out_res,  # 输出的X轴方向上的分辨率
        yRes=out_res,  # 输出的y轴方向上的分辨率
        dstNodata=nodata_value,  # 栅格矩阵中的无效值
        outputType=out_type,  # 输出的数据类型
        multithread=True,  # 多线程处理
        resampleAlg=resamples[resample]  # 重采样分辨率
    )
    dst_ds = gdal.Warp(out_path, src_ds, options=options)

    if dst_ds:  # 释放缓存和资源
        dst_ds.FlushCache()
        src_ds, dst_ds = None, None


def cal_zone(xx, yy, proj_from, proj_to):
    transformer = Transformer.from_proj(proj_from, proj_to, always_xy=True)  # 特别关注这个always_xy表示,需要显示True
    lon, lat = transformer.transform(xx, yy)  # 将proj_from投影的角点坐标转化为proj_to的坐标
    lon_min, lon_max = min(lon), max(lon)
    central_lon = (lon_min + lon_max) / 2
    zone = ceil((central_lon + 180) / 6)

    return zone


def extract2d_lons_lats(img_path):
    """
    基于当前tiff文件的地理参数分别构建经纬度数据集(2D)(其实使用rasterio.transform.xy可以快速解决)
    :param img_path: 待提取经纬度数据集的影像文件路径
    :return: 返回二维的lon和lats
    """

    lon_edges, lat_edges = extract1d_lons_lats(img_path)
    lons, lats = np.meshgrid(lon_edges, lat_edges)

    return lons, lats


def extract1d_lons_lats(img_path):
    """
    基于当前tiff文件的地理参数分别构建经纬度数据集(2D)
    :param img_path: 待提取经纬度数据集的影像文件路径
    :return: 返回一维的lons和lats
    """

    raster = gdal.Open(img_path)
    # 获取仿射变换参数和基本栅格信息
    [lon_min, lon_res, _, lat_max, _, minus_lat_res] = raster.GetGeoTransform()
    lat_res = -minus_lat_res
    row, col = raster.RasterYSize, raster.RasterXSize
    lon_max = lon_min + col * lon_res
    lat_min = lat_max + row * minus_lat_res

    # 构建经纬度数据集
    lon_edges = np.arange(lon_min + lon_res / 2, lon_max, lon_res)  # +半个分辨率是表示生成像元中心位置的经纬度
    lat_edges = np.arange(lat_min + lat_res / 2, lat_max, lat_res)

    return lon_edges, lat_edges


def write_tiff(img_arr, dst_path, template_path=None, geo_transform=None, nodata_value=None, dtype=gdal.GDT_Float32):
    """
    输出单波段的栅格矩阵输出为Geotiff文件
    :param img_arr: 栅格矩阵
    :param dst_path: 输出路径
    :param template_path: 模板文件, 用于获取地理参数
    :param geo_transform: 地理仿射参数,默认WGS84
    :param nodata_value: 无效值设置
    :param dtype: 存储的数据类型, 默认浮点型
    :return: None
    """

    row, col = img_arr.shape
    driver = gdal.GetDriverByName('GTiff')  # 获取创建geotiff的驱动器
    img = driver.Create(dst_path, col, row, 1, dtype)  # (输出路径, x轴范围, y轴范围, 波段数, 数据类型)
    b1 = img.GetRasterBand(1)

    # 设置地理参数
    if template_path is not None:
        template_ds = gdal.Open(template_path)
        img.SetGeoTransform(template_ds.GetGeoTransform())
        img.SetProjection(template_ds.GetProjection())
    elif geo_transform is not None:
        img.SetProjection('EPSG:4326')  # 设置默认坐标系为WGS84
        img.SetGeoTransform(geo_transform)  # 设置仿射系数

    # 写入栅格矩阵
    b1.WriteArray(img_arr)

    # 其他设置
    if nodata_value is not None:
        b1.SetNoDataValue(nodata_value)
    b1.ComputeStatistics(False)

    img = None


def cal_wind_direction(u, v):
    # 计算来向向量 (-u, -v) 的数学角度 -- (来向向量: 风吹来的方向, 例如北风是从北边吹来的,所以来向向量的方向是正北)
    wind_dir_rad = np.arctan2(-v, -u)  # 参数顺序为 y=-v, x=-u
    """
    这里是对于np.arctan2(y, x)是传入一个向量(x, y), 计算其与向量(1, 0)(该向量与X轴正方向一致,此处坐标均为数学坐标系/二维笛卡尔坐标系)的夹角,
    夹角范围为(-180, 180), 若为一二象限即是正,若为三四象限即为负,这与数学上的逆时针为正角顺时针为负角一致.
    此外,
    此处的u和v,首先解释u表示风吹向东边的速度;v表示风吹向北边的速度;(负号表示与定义方向相反)
    由上面可以知道, u表示x轴方向上的分速度, v表示y轴方向上的分速度;
    基于u和v可以计算出风运动的方向和速度,速度这里我们暂时不管,但是对于风运动的方向,显然与向量(u, v)的方向一致;
    但是风运动的方向和向量(u, v)的方向都是表示风往哪个方向吹或者风要吹到哪里去;这与风向的定义正好相反,例如:
    北风表示从北边吹来的风,所以是吹往南边的风,假定风速是1m/s,那么u=0,v=-1m/s(由v的定义知往南吹与定义方向相反所以添上负号).
    如果直接传入arctan2(-1, 0),那么实际上得到的是向量(0, -1)的方向与向量(1, 0)方向上的夹角.
    但是我们应该是需要得到向量(0, 1)的方向与向量(1, 0)方向上的夹角.
    大家自行体会为什么需要添上负号,本质就是u和v定义正方向是风吹向的方向,而风向定义的方向是风来时或者从哪里吹来的方向--方向刚好相反
    """

    # 将弧度转换为度数 [-pi, pi] ==> [-180, 180]
    wind_dir_deg = np.degrees(wind_dir_rad)

    # 转换为地理角度(正北为0°，顺时针旋转)
    wind_dir_geo = (90 - wind_dir_deg) % 360
    """
    根据arctan2的定义知,其输出是指风向向量(当你输入是-u, -v时符合风向定义)与向量(1, 0)方向或者说x轴正方向之间的夹角
    但是风向的定义北风为0°,如此顺时针旋转到360度回到正北.两个夹角的定义不相同, 需要进行换算
    """

    return wind_dir_geo


def cal_wind_slope_angle(out_path, u10_path, v10_path, aspect_path, slope_path):
    """
    基于u10和v10计算风向, 基于风向和坡向、坡度计算风坡夹角
    风坡夹角: 风向与坡向夹角的余弦值和坡度正弦值的乘积
    :param out_path: 风坡夹角的输出路径
    :param u10_path: u10的输入路径
    :param v10_path: v10的输入路径
    :param aspect_path: 坡向的输入路径
    :param slope_path: 坡度的输入路径
    :return: None
    """

    # 读取经纬向风速,坡向坡度
    with rio.open(u10_path) as f:
        u10 = f.read(1, masked=True)  # 读取第一个波段
        meta = f.meta
        """
        .meta返回当前tiff文件的元数据, 包括格格式(GTiff)、数据类型(dtype)、无效值(nodata)、行列数和波段数(width,height,count),
        坐标参考系(crs)、仿射参数(transform)
        示例:
        {'driver': 'GTiff',
         'dtype': 'float32',
         'nodata': nan,
         'width': 129,
         'height': 133,
         'count': 1,
         'crs': CRS.from_wkt('GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6367470,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'),
         'transform': Affine(0.1, 0.0, 97.30000000000001,
                0.0, -0.1, 34.4)}
        """
    with rio.open(v10_path) as f:
        v10 = f.read(1, masked=True)
    with rio.open(aspect_path) as f:
        aspect = f.read(1, masked=True)
    with rio.open(slope_path) as f:
        slope = f.read(1, masked=True)

    # 计算风向
    wind_dir = cal_wind_direction(u10, v10)
    # 计算风向和坡向的夹角
    theta_diff = np.abs(wind_dir - aspect)
    theta = np.minimum(theta_diff, 360 - theta_diff)
    # 计算风坡夹角
    ws_angle = np.cos(np.radians(theta)) * np.sin(np.radians(slope))

    with rio.open(out_path, 'w', **meta) as f:
        f.write(ws_angle, 1)
