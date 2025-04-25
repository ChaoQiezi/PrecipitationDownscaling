# @Author  : ChaoQiezi
# @Time    : 2025/4/6 上午10:34
# @Email   : chaoqiezi.one@qq.com
# @FileName: dem_preprocessing

"""
This script is used to 将30m分辨率的SRTM DEM的tiff文件输出为0.1°和1km分辨率,此外基于DEM计算经纬度、坡度坡向、风坡夹角等地理因子

注意: 此处的1km分辨率实际上是指输出为地理坐标系(WGS84坐标系）的0.009°
"""

import os
import numpy as np

from Src.utils import clip_mask, extract2d_lons_lats, write_tiff
import Config


# 准备
src_dem_path = r'E:\SoftwaresStorage\ArcGISPro_Storage\Projects\PrecipitationDownscaling\DEM\sw_china_dem.tif'
# dem_01deg_path = r'E:\SoftwaresStorage\ArcGISPro_Storage\Projects\PrecipitationDownscaling\DEM\sw_china_dem_0.1deg.tif'
# dem_1km_path = r'E:\SoftwaresStorage\ArcGISPro_Storage\Projects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif'
dem_01deg_path = r'E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_0.1deg.tif'
dem_1km_path = r'E:\Datasets\Objects\PrecipitationDownscaling\DEM\sw_china_dem_1km.tif'
out_lon_lat_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\LonLat'
out_slope_dir = r'E:\Datasets\Objects\PrecipitationDownscaling\Slope'

# DEM处理
clip_mask(src_dem_path, Config.region_path, dem_01deg_path, out_res=0.1)
clip_mask(src_dem_path, Config.region_path, dem_1km_path, out_res=0.009)

# 经纬度处理
# 0.1°
lons, lats = extract2d_lons_lats(dem_01deg_path)
temp_lon_path = os.path.join(out_lon_lat_dir, 'lon_0.1deg_temp.tif')
temp_lat_path = os.path.join(out_lon_lat_dir, 'lat_0.1deg_temp.tif')
write_tiff(lons, temp_lon_path, dem_01deg_path)
write_tiff(lats, temp_lat_path, dem_01deg_path)
out_lon_path = os.path.join(out_lon_lat_dir, 'lon_0.1deg.tif')
out_lat_path = os.path.join(out_lon_lat_dir, 'lat_0.1deg.tif')
clip_mask(temp_lon_path, Config.region_path, out_lon_path, out_res=0.1, remove_src=True, dst_nodata=np.nan)
clip_mask(temp_lat_path, Config.region_path, out_lat_path, out_res=0.1, remove_src=True, dst_nodata=np.nan)
# 1km
lons, lats = extract2d_lons_lats(dem_1km_path)
temp_lon_path = os.path.join(out_lon_lat_dir, 'lon_1km_temp.tif')
temp_lat_path = os.path.join(out_lon_lat_dir, 'lat_1km_temp.tif')
write_tiff(lons, temp_lon_path, dem_1km_path)
write_tiff(lats, temp_lat_path, dem_1km_path)
out_lon_path = os.path.join(out_lon_lat_dir, 'lon_1km.tif')
out_lat_path = os.path.join(out_lon_lat_dir, 'lat_1km.tif')
clip_mask(temp_lon_path, Config.region_path, out_lon_path, out_res=0.009, remove_src=True, dst_nodata=np.nan)
clip_mask(temp_lat_path, Config.region_path, out_lat_path, out_res=0.009, remove_src=True, dst_nodata=np.nan)
# 坡度-使用arcgis pro完成
# 坡向-使用arcgis pro完成

print('DEM处理完成.')
