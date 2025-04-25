# @Author  : ChaoQiezi
# @Time    : 2025/4/6 下午7:00
# @Email   : chaoqiezi.one@qq.com
# @FileName: mod13a3_processing

"""
This script is used to 对MOD13A3 植被指数数据集进行预处理

包括: HDF4转tiff、镶嵌、裁剪、掩膜
"""

import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from glob import glob

from Src.utils import img_mosaic, img_warp, cal_zone, clip_mask
import Config


# 准备
in_dir = r'G:\MOD13A3'
out_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\NDVI"
var_name = '1 km monthly NDVI'
res_folder_name = '0.1deg'
out_res = 0.1
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
out_dir = os.path.join(out_dir, res_folder_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 镶嵌
rd = relativedelta(end_date, start_date)
months = rd.years * 12 + rd.months + 1
for month_count in range(months):
    # 检索当前日期的数据集
    cur_date = (start_date + relativedelta(months=month_count)).timetuple()
    wildcard = 'MOD13A3.A{}{:03}*.hdf'.format(cur_date.tm_year, cur_date.tm_yday)
    mosaic_paths = glob(os.path.join(in_dir, wildcard))
    if len(mosaic_paths) == 0:
        print('当前日期({})无数据集'.format(cur_date))

    # 镶嵌
    mosaic_img, geo_transform, sinu_proj4, [xx, yy] = img_mosaic(mosaic_paths, '1 km monthly NDVI',
                                                                 unit_conversion=True,
                                                                 scale_factor_op='divide', mosaic_mode='max')
    # # 重投影(sinu正弦投影转换为UTM-横轴墨卡托)
    # reproj_path = os.path.join(out_dir, 'mosaic.tif')
    # zone = cal_zone(xx, yy, sinu_proj4, 'EPSG: 4326')
    # out_epsg = 32600 + zone
    # img_warp(mosaic_img, reproj_path, geo_transform, sinu_proj4, dst_epsg=out_epsg)

    # 重投影(sinu正弦投影转换为WGS84)
    reproj_path = os.path.join(out_dir, 'mosaic.tif')
    img_warp(mosaic_img, reproj_path, geo_transform, sinu_proj4, dst_epsg=4326)

    # 裁剪掩膜
    out_filename = 'MOD13A3_{}_{}_{:02}_{}.tif'.format(var_name.rsplit(' ', 1)[-1], cur_date.tm_year, cur_date.tm_mon, res_folder_name)
    out_path = os.path.join(out_dir, out_filename)
    clip_mask(reproj_path, Config.region_path, out_path, out_res=out_res, remove_src=True)
    print('处理: {}'.format(out_filename))
print('数据集处理完毕.')