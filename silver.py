from silver_lib import generate_pic_pair_vids, cloud_point_pixel_pairs
from namelist import Velox_NetCDF_File

if __name__ == '__main__':
    pic_pair_vids_list = generate_pic_pair_vids(Velox_NetCDF_File)
    pixel_pairs_list = cloud_point_pixel_pairs(pic_pair_vids_list)