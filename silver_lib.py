from pathlib import Path
import xarray as xr
import cv2 as cv
import numpy as np


def generate_pic_pair_vids(Velox_NetCDF_File):
    '''
    Function that generates image pairs from Velox brightness temperature data. The image pairs are each one second
    apart and are saved as grayscale videos in the vids_tmp folder.

    :param Velox_NetCDF_File: Path to Velox NetCDF File
    :return: True
    '''
    Path('vids_tmp').mkdir(exist_ok=True)
    velox_data = xr.open_dataset(Velox_NetCDF_File)
    time = velox_data['time']
    BT_2D = velox_data['BT_2D']
    vid_size = np.shape(BT_2D)[1:]
    for t in range(len(time)-1):
        if time[t+1]-time[t] == np.timedelta64(1, 's'):
            vid_output = out = cv.VideoWriter('vids_tmp/'+str(t)+'.avi',
                                              cv.VideoWriter_fourcc(*'XVID'),
                                              1,
                                              vid_size,
                                              isColor=False)
            vmax = np.max(np.stack((BT_2D[t+1], BT_2D[t])))
            vmin = np.min(np.stack((BT_2D[t+1], BT_2D[t])))
            grayscale_data0 = np.array(np.round((BT_2D[t] - vmin) / (vmax - vmin) * 255, 0), dtype=np.uint8)
            grayscale_data1 = np.array(np.round((BT_2D[t+1] - vmin) / (vmax - vmin) * 255, 0), dtype=np.uint8)
            vid_output.write(grayscale_data0.T[::-1])
            vid_output.write(grayscale_data1.T[::-1])
            vid_output.release()
    return True