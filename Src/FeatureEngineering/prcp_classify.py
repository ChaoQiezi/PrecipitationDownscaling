# @Author  : ChaoQiezi
# @Time    : 2025/4/26 上午12:29
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: prcp_classify

"""
This script is used to 针对降水数据进行降水分级/分类,以tiff文件输出分类结果(也可以使用arcgis模型构建器实现)

摘自: 中国气象局 -【气象科普】降水的等级划分
url: https://www.cma.gov.cn/2011xzt/2012zhuant/20120928_1_1_1_1/2010052703/201212/t20121212_195616.html
分级:
    无雨或小雨(栅格值: 0): < 10 mm/day
    中雨(栅格值: 1): 10 ~ 25 mm/day
    大雨(栅格值: 2): 25 ~ 50 mm/day
    暴雨(栅格值: 3): > 50 mm/day

上述无效, 因为当前降水数据集是月尺度的, 尽管原始单位是mm/hr,但是这是求取平均之后,实际并不能反应当天的降水量,只是反应一段时间的降水平均情况,
所以不使用原始阿mm/hr分级,而是使用月累计降水量进行分级:
 (0, 20]
 [20, 50)
 [50, 100)
 (100, ∞)
"""

import os
import rasterio as rio
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr

import Config
from Src.utils import write_tiff

# 准备
# in_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG\mm_hr\0.1deg'
in_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\GPM_IMERG\0.1deg'
out_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\prcp_class'
chart_dir = r'F:\PyProJect\PrecipitationDownscaling\Asset\chart'
deg_name = '0.1deg'
km_name = '1km'
out_deg_dir = os.path.join(out_dir, deg_name)
out_km_dir = os.path.join(out_dir, km_name)
for cur_dir in [out_deg_dir, out_km_dir]:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
gpm_paths = glob(os.path.join(in_dir, 'GPM*.tif'))

# 迭代分级处理
for cur_path in gpm_paths:
    # 读取降水
    with rio.open(cur_path) as src:
        cur_prcp = src.read(1, masked=True)
    # 分级
    cur_prcp_class = np.full(cur_prcp.shape, -9999, dtype=np.int32)
    cur_prcp_class[cur_prcp <= 20] = 0
    cur_prcp_class[(cur_prcp > 20) & (cur_prcp <= 50)] = 1
    cur_prcp_class[(cur_prcp > 50) & (cur_prcp <= 100)] = 2
    cur_prcp_class[(cur_prcp > 100)] = 3
    # 输出为0.1deg的分级Geotiff
    cur_filename = os.path.basename(cur_path)  # GPM_2019_01_0.1deg.tif
    out_deg_filename = 'prcp_class_' + cur_filename.split('_', 1)[1]
    out_deg_path = os.path.join(out_deg_dir, out_deg_filename)
    with rio.open(cur_path) as src:
        meta = src.meta.copy()
    meta.update(dtype='int16', nodata=-9999)
    with rio.open(out_deg_path, 'w', **meta) as src:
        src.write(cur_prcp_class, 1)
    # 重采样为1km(约0.009°)
    with ogr.Open(Config.region_path) as src:
        min_x, max_x, min_y, max_y = src.GetLayer().GetExtent()
    options = gdal.WarpOptions(
        format='GTiff',
        xRes=0.009,
        yRes=0.009,
        outputBounds=[min_x, min_y, max_x, max_y],
        resampleAlg=gdal.GRA_Cubic,  # 重采样方法
        targetAlignedPixels=True  # 对齐像元
    )
    # 输出1km的分级Geotiff
    out_km_filename = out_deg_filename.rsplit('_', 1)[0] + '_' + km_name + '.tif'
    out_km_path = os.path.join(out_km_dir, out_km_filename)
    gdal.Warp(out_km_path, out_deg_path, options=options)

    # plt.hist(cur_prcp.flatten(), bins=100)
    # plt.savefig(os.path.join(chart_dir, cur_filename.split('.')[0] + '.png'))
    # plt.close()

    print('处理: {}, max: {:.2f}, min: {:.2f}'.format(out_km_filename, cur_prcp.max(), cur_prcp.min()))
print('处理完成.')
