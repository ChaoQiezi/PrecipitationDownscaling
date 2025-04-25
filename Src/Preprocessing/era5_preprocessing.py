# @Author  : ChaoQiezi
# @Time    : 2025/4/22 上午1:11
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: era5_preprocessing

"""
This script is used to 处理ERA5-u10和v10

包括: nc转tiff、裁剪、掩膜、重采样
"""

import os
from glob import glob

from Src.utils import nc2tiff, clip_mask
import Config

# 准备
in_dir = r'G:\ERA5'
out_dir = r'E:\Datasets\Objects\PrecipitationDownscaling'
var_names = {
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10'
}
out_res = 0.009
res_folder_name = '1km'

# 迭代处理
for var_name, out_var_name in var_names.items():  # 处理不同变量的nc文件
    cur_in_dir = os.path.join(in_dir, var_name)
    cur_var_paths = glob(os.path.join(cur_in_dir, '{}*.nc'.format(var_name)))  # 检索当前变量下的nc文件
    cur_var_dir = os.path.join(out_dir, out_var_name, res_folder_name)
    if not os.path.exists(cur_var_dir):
        os.makedirs(cur_var_dir)
    for cur_path in cur_var_paths:  # 循环当前变量下载的所有nc文件
        cur_filename = os.path.basename(cur_path).rsplit('.', 1)[0]
        cur_filename = out_var_name + '_' + cur_filename.rsplit('_', 1)[1]
        cur_temp_path = os.path.join(cur_var_dir, cur_filename + '_temp.tif')
        cur_out_path = os.path.join(cur_var_dir, cur_filename + '_{}.tif'.format(res_folder_name))

        # nc转tiff
        nc2tiff(cur_path, cur_temp_path, out_var_name)
        # 裁剪掩膜
        clip_mask(cur_temp_path, Config.region_path, cur_out_path, out_res=out_res, remove_src=True)
        print('处理: {}'.format(os.path.basename(cur_out_path)))
print('处理完成.')
