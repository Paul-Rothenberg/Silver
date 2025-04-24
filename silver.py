from silver_lib import generate_pic_pair_vids, cloud_point_pixel_pairs, stereographic_reconstruction, save_to_NetCDF
from namelist import (Velox_BT_File, Velox_VDC_File, HALO_IRS_File, MNT_File, vid_edge_trim, ERA5_UV_Wind_File,
                      DSM_file, Pcs_save_path, process_number)

if __name__ == '__main__':
    pic_pair_vids_list = generate_pic_pair_vids(Velox_BT_File, vid_edge_trim)
    pixel_pairs_list = cloud_point_pixel_pairs(pic_pair_vids_list)
    cloud_point_main_storage = stereographic_reconstruction(pic_pair_vids_list, pixel_pairs_list, Velox_VDC_File,
                                                            HALO_IRS_File, MNT_File, vid_edge_trim, ERA5_UV_Wind_File,
                                                            DSM_file, process_number)
    save_to_NetCDF(cloud_point_main_storage, Pcs_save_path)