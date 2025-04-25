# @Author  : ChaoQiezi
# @Time    : 2025/4/8 下午8:18
# @Email   : chaoqiezi.one@qq.com
# @FileName: mod11a2_processing

"""
This script is used to 对MOD11A2 植被指数数据集进行预处理

包括: HDF4转tiff、镶嵌、裁剪、掩膜
"""

import calendar
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from glob import glob

from Src.utils import img_mosaic, img_warp, cal_zone, clip_mask
import Config


# 准备
in_dir = r'G:\MOD11A2'
out_dir = r"E:\Datasets\Objects\PrecipitationDownscaling\LST_Night"
var_name = 'LST_Night_1km'
res_folder_name = '0.1deg'
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
out_res = 0.1
out_dir = os.path.join(out_dir, res_folder_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 镶嵌
rd = relativedelta(end_date, start_date)
for year_count in range(rd.years + 1):

    cur_year_start_date = start_date + relativedelta(years=year_count)
    if calendar.isleap(cur_year_start_date.year):
        year_end_day = 366
    else:
        year_end_day = 365

    cur_year_paths = {}
    for cycle_count in range(year_end_day // 8):
        cur_date = (cur_year_start_date + relativedelta(days=cycle_count * 8)).timetuple()
        # 检索当前日期的数据集
        wildcard = 'MOD11A2.A{}{:03}*.hdf'.format(cur_date.tm_year, cur_date.tm_yday)
        mosaic_paths = glob(os.path.join(in_dir, wildcard))
        if len(mosaic_paths) == 0:
            print('当前日期({})无数据集'.format(cur_date))
            continue

        # 添加当前循环下检索的路径
        add_paths = cur_year_paths.get(cur_date.tm_mon, [])
        add_paths.extend(mosaic_paths)
        cur_year_paths[cur_date.tm_mon] = add_paths

    for cur_month, cur_month_paths in cur_year_paths.items():
        cur_date = cur_year_start_date + relativedelta(month=cur_month)
        # 镶嵌
        mosaic_img, geo_transform, sinu_proj4, [xx, yy] = img_mosaic(cur_month_paths, var_name,
                                                                     unit_conversion=True,
                                                                     scale_factor_op='multiply', mosaic_mode='max')
        # 开尔文换算摄氏度
        mosaic_img = mosaic_img - 273.15

        # # 重投影(sinu正弦投影转换为UTM-横轴墨卡托)
        # reproj_path = os.path.join(out_dir, 'mosaic.tif')
        # zone = cal_zone(xx, yy, sinu_proj4, 'EPSG: 4326')
        # out_epsg = 32600 + zone
        # img_warp(mosaic_img, reproj_path, geo_transform, sinu_proj4, out_res=1000, dst_epsg=out_epsg)

        # 重投影(sinu正弦投影转换为WGS84)
        reproj_path = os.path.join(out_dir, 'mosaic.tif')
        img_warp(mosaic_img, reproj_path, geo_transform, sinu_proj4, dst_epsg=4326)

        # 裁剪掩膜
        out_filename = 'MOD11A2_{}_{}_{:02}_{}.tif'.format(var_name.rsplit('_', 1)[0], cur_date.year, cur_date.month, res_folder_name)
        out_path = os.path.join(out_dir, out_filename)
        clip_mask(reproj_path, Config.region_path, out_path, out_res=out_res, remove_src=True)
        print('处理: {}-{}'.format(var_name, out_filename))
print('数据集处理完毕.')
