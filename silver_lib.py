from pathlib import Path
import xarray as xr
import cv2 as cv
import numpy as np


def generate_pic_pair_vids(Velox_NetCDF_File):
    '''
    Function that generates image pairs from Velox brightness temperature data. The image pairs are each one second
    apart and are saved as grayscale videos in the vids_tmp folder.

    :param Velox_NetCDF_File: Path to Velox NetCDF File
    :return pic_pair_vids_list: List of Paths from image pair videos
    '''
    Path('vids_tmp').mkdir(exist_ok=True)
    pic_pair_vids_list = []
    velox_data = xr.open_dataset(Velox_NetCDF_File)
    time = velox_data['time']
    BT_2D = velox_data['BT_2D']
    vid_size = np.shape(BT_2D)[1:]
    for t in range(len(time)-1):
        if time[t+1]-time[t] == np.timedelta64(1, 's'):
            vid_output = cv.VideoWriter('vids_tmp/'+str(time[t].values)+'.avi',
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
            pic_pair_vids_list.append('vids_tmp/'+str(time[t].values)+'.avi')
    return pic_pair_vids_list


def cloud_point_pixel_pairs(pic_pair_vids_list):
    '''
    Function that searches for identical cloud points in consecutive images. It is based on Shi-Tomasi corner detection
    and calculates the optical flow using the pyramid implication of the Lucas-Kanade algorithm. It also checks the
    quality of the point pairs by a backward calculation of the optical flow, returning only high quality points.

    :param pic_pair_vids_list: List of Paths from image pair videos
    :return pixel_pairs_list: List of Arrays containing the image point pairs in pixel units
    '''
    pixel_pairs_list = []
    for pic_pair_vid in pic_pair_vids_list:
        cap = cv.VideoCapture(pic_pair_vid)

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # read in frames
        ret, old_frame = cap.read()
        ret, new_frame = cap.read()
        # convert to grayscale (since the images are already in grayscale, only the array structure is adjusted)
        old_gray_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        new_gray_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        # select good corner points
        p0 = cv.goodFeaturesToTrack(old_gray_frame, mask=None, **feature_params)
        # calculate the optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray_frame, new_gray_frame, p0, None, **lk_params)
        if p1 is not None:
            # check quality of point pairs by calculating the optical flow backwards
            p0r, _st, _err = cv.calcOpticalFlowPyrLK(new_gray_frame, old_gray_frame, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            # maximum deviation of p0 and p0r should be less than one pixel in x- and y-direction
            good = d < 1
            good_old_points = p0[(st == 1) & (_st == 1) & (good.reshape(-1, 1))]
            good_new_points = p1[(st == 1) & (_st == 1) & (good.reshape(-1, 1))]

            pixel_pairs = np.stack((good_old_points, good_new_points), axis=1)
        else:
            pixel_pairs = None
        pixel_pairs_list.append(pixel_pairs)
    return pixel_pairs_list