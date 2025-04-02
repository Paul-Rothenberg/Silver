from pathlib import Path
import xarray as xr
import cv2 as cv
import numpy as np
import mounttree as mnt


def generate_pic_pair_vids(Velox_BT_File, vid_edge_trim):
    '''
    Function that generates image pairs from Velox brightness temperature data. The image pairs are each one second
    apart and are saved as grayscale videos in the vids_tmp folder.

    :param Velox_BT_File: Path to Velox NetCDF file containing the brightness temperatures
    :return: List of Paths from image pair videos
    '''
    Path('vids_tmp').mkdir(exist_ok=True)
    pic_pair_vids_list = []
    velox_data = xr.open_dataset(Velox_BT_File)
    time = velox_data['time']
    BT_2D = velox_data['BT_2D']
    # Removing pixel rows or columns if the number of pixels is odd. Data is removed from the edges with the smaller
    # vid_edge_trim.
    BT_2D = BT_2D.isel({'y': range((vid_edge_trim[3]<vid_edge_trim[1] and sum(vid_edge_trim[1::2])%2!=0),
                                   640-sum(vid_edge_trim[1::2])-
                                   (vid_edge_trim[3]>vid_edge_trim[1] and sum(vid_edge_trim[1::2])%2!=0)),
                        'x': range((vid_edge_trim[0]>vid_edge_trim[2] and sum(vid_edge_trim[0::2])%2!=0),
                                   512-sum(vid_edge_trim[0::2])-
                                   (vid_edge_trim[0]<vid_edge_trim[2] and sum(vid_edge_trim[0::2])%2!=0))})
    # trims the edges to an even number of pixels so that no resampling takes place during video generation
    vid_size = (640-sum(vid_edge_trim[1::2])-(sum(vid_edge_trim[1::2])%2!=0),
                512-sum(vid_edge_trim[0::2])-(sum(vid_edge_trim[0::2])%2!=0))
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
    :return: List of Arrays containing the image point pairs in pixel units
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


def viewing_direction(pixel_pairs, Velox_VDC_Data, vid_edge_trim):
    '''
    This function translates an array of pixel position pairs into an array of viewing direction vectors. First, it
    calculates how the calibration data set is positioned in relation to the brightness temperature data set. Then a
    decision is made as if the zenith and azimuth angle of a point needs to be interpolated. This is the case if the
    point does not lie directly on a pixel. For interpolation, a weighted average is calculated from the four
    surrounding pixels. The distance is included linearly in the weights, whereby the influence decreases to zero over a
    distance of one pixel. In the next step, the viewing direction vector is calculated from the zenith and azimuth
    angle found. If there are no four surrounding pixels during the interpolation, a None vector is added. This is only
    the case if the point lies between the center of an edge pixel and the absolute edge of the image and the trim of
    this side is zero. If the translated array contains None vectors, the entire pixel point pair is deleted and only
    valid vectors are returned.

    :param pixel_pairs: array containing the pixel point pairs of consecutive images
    :param Velox_VDC_Data: Velox calibration file which assigns a viewing direction to each pixel in the camera
                           reference frame
    :param vid_edge_trim: list specifying how the brightness temperature data set was trimmed in respect to the raw data
    :return: array which contains the translated viewing vectors in the camera reference frame
    '''
    # trimming the calibration data set to the video size
    real_vid_edge_trim = [vid_edge_trim[0]+(vid_edge_trim[0]<vid_edge_trim[2] and sum(vid_edge_trim[0::2])%2!=0),
                          vid_edge_trim[1]+(vid_edge_trim[1]<vid_edge_trim[3] and sum(vid_edge_trim[1::2])%2!=0),
                          vid_edge_trim[2]+(vid_edge_trim[2]<vid_edge_trim[0] and sum(vid_edge_trim[0::2])%2!=0),
                          vid_edge_trim[3]+(vid_edge_trim[3]<vid_edge_trim[1] and sum(vid_edge_trim[1::2])%2!=0)]
    real_vid_edge_trim = [0,0,0,0] # ToDo: Remove when correct calibration data is ready
    # This line is used for debugging purposes only. Data is not selected by using coordinates but by indices.
    Velox_VDC_Data = Velox_VDC_Data.assign_coords({'x-pixel': range(0-real_vid_edge_trim[3],
                                                                    640-real_vid_edge_trim[3]-5), # ToDo: Remove -5 when correct calibration data is ready
                                                   'y-pixel': range(0-real_vid_edge_trim[0],
                                                                    512-real_vid_edge_trim[0]-5)})# ToDo: Remove -5 when correct calibration data is ready
    zenith = Velox_VDC_Data['zenith']
    azimuth = Velox_VDC_Data['azimuth']

    VD_Vector_storage = []
    for pixel in pixel_pairs.reshape(-1,2):
        if pixel[0]%1 == 0 and pixel[1]%1 == 0:
            x = int(pixel[0])+real_vid_edge_trim[3]
            y = int(pixel[1])+real_vid_edge_trim[0]
            z = zenith[x][y].values*np.pi/180
            a = azimuth[x][y].values*np.pi/180
            VD_Vector = (np.tan(z)*np.cos(a), -np.tan(z)*np.sin(a), 1)
            VD_Vector_storage.append(VD_Vector)
        else:
            x0 = int(np.floor(pixel[0]))+real_vid_edge_trim[3]
            y0 = int(np.floor(pixel[1]))+real_vid_edge_trim[0]
            if x0 < 0 or y0 < 0:
                VD_Vector_storage.append((None, None, None))
                continue
            z_patch = zenith[x0:x0+2].T[y0:y0+2].T
            a_patch = azimuth[x0:x0+2].T[y0:y0+2].T
            if z_patch.shape != (2, 2) or a_patch.shape != (2, 2):
                VD_Vector_storage.append((None, None, None))
                continue
            z_patch = z_patch.values*np.pi/180
            a_patch = a_patch.values*np.pi/180
            d = np.array([[((pixel[0]-x0)**2+(pixel[1]-y0)**2)**0.5,
                           ((pixel[0]-x0)**2+(1-pixel[1]+y0)**2)**0.5],
                          [((1-pixel[0]+x0)**2+(pixel[1]-y0)**2)**0.5,
                           ((1-pixel[0]+x0)**2+(1-pixel[1]+y0)**2)**0.5]])
            w = np.vectorize(lambda x: max(0, 1-x), otypes=[float])(d)
            w = w/np.sum(w)
            # circularity consideration of the zenith angle data not necessary since the breakpoint is behind the camera
            z = np.sum(z_patch*w)
            # circularity consideration of the azimuth angle data
            a = np.arctan2(np.sum(np.sin(a_patch)*w), np.sum(np.cos(a_patch)*w))
            VD_Vector = (np.tan(z)*np.cos(a), -np.tan(z)*np.sin(a), 1)
            VD_Vector_storage.append(VD_Vector)
    VD_Vector_array = np.array(VD_Vector_storage).reshape(-1,2,3)
    VD_Vector_clean = (np.delete(VD_Vector_array, np.unique(np.where(VD_Vector_array == None)[0]), 0)
                       .astype(np.float64))
    return VD_Vector_clean


def stereographic_reconstruction(pic_pair_vids_list, pixel_pairs_list, Velox_VDC_File, HALO_IRS_File, MNT_File,
                                 vid_edge_trim):
    coordinate_systems = mnt.load_mounttree(MNT_File)
    Velox_VDC_Data = xr.open_dataset(Velox_VDC_File)
    HALO_IRS_Data = xr.open_dataset(HALO_IRS_File)
    IRS_TIME = HALO_IRS_Data['time']    # time
    IRS_LAT = HALO_IRS_Data['IRS_LAT'] # Latitude
    IRS_LON = HALO_IRS_Data['IRS_LON'] # Longitude
    IRS_ALT = HALO_IRS_Data['IRS_ALT'] # Altitude
    IRS_PHI = HALO_IRS_Data['IRS_PHI'] # Roll
    IRS_THE = HALO_IRS_Data['IRS_THE'] # Pitch
    IRS_HDG = HALO_IRS_Data['IRS_HDG'] # Yaw

    for vid in range(len(pic_pair_vids_list)): # ToDo: Remove -56
        vid_time = pic_pair_vids_list[vid][-33:-4]
        time_index = np.where(IRS_TIME.astype(str) == vid_time)[0].item()
        # reconstruction doesn't work if P1=P2
        if (IRS_LAT[time_index] == IRS_LAT[time_index+1] and IRS_LON[time_index] == IRS_LON[time_index+1] and
                IRS_ALT[time_index] == IRS_ALT[time_index+1]):
            continue
        EARTH_frame = coordinate_systems.get_frame('EARTH')
        coordinate_systems.update(lat=IRS_LAT[time_index], lon=IRS_LON[time_index], height=IRS_ALT[time_index],
                                  roll=IRS_PHI[time_index], pitch=IRS_THE[time_index], yaw=IRS_HDG[time_index])
        VE_transformation = coordinate_systems.get_transformation('VELOX', 'EARTH')
        P1E = VE_transformation.apply_point(0, 0, 0)
        coordinate_systems.update(lat=IRS_LAT[time_index+1], lon=IRS_LON[time_index+1], height=IRS_ALT[time_index+1],
                                  roll=IRS_PHI[time_index+1], pitch=IRS_THE[time_index+1], yaw=IRS_HDG[time_index+1])
        VE_transformation = coordinate_systems.get_transformation('VELOX', 'EARTH')
        P2E = VE_transformation.apply_point(0, 0, 0)
        PrefE = [(a+b)/2 for a, b in zip(P1E, P2E)]
        PrefE_N = EARTH_frame.toNatural(PrefE)
        ref_lat, ref_lon, ref_height = PrefE_N

        if np.array_equal(pixel_pairs_list[vid], None):
            continue # ToDo: adapting behavior depending on data storage
        view_vectors = viewing_direction(pixel_pairs_list[vid], Velox_VDC_Data, vid_edge_trim)
        if view_vectors.shape == (0, 2, 3):
            continue # ToDo: adapting behavior depending on data storage

        coordinate_systems.update(lat=IRS_LAT[time_index], lon=IRS_LON[time_index], height=IRS_ALT[time_index],
                                  roll=IRS_PHI[time_index], pitch=IRS_THE[time_index], yaw=IRS_HDG[time_index],
                                  ref_lat=ref_lat, ref_lon=ref_lon, ref_height=ref_height)
        VS_transformation = coordinate_systems.get_transformation('VELOX', 'Stereo')
        P1 = np.array(VS_transformation.apply_point(0, 0, 0))
        VV1 = np.stack(VS_transformation.apply_direction(*view_vectors[:,0,:].T)).T

        coordinate_systems.update(lat=IRS_LAT[time_index+1], lon=IRS_LON[time_index+1], height=IRS_ALT[time_index+1],
                                  roll=IRS_PHI[time_index+1], pitch=IRS_THE[time_index+1], yaw=IRS_HDG[time_index+1])
        VS_transformation = coordinate_systems.get_transformation('VELOX', 'Stereo')
        P2 = np.array(VS_transformation.apply_point(0, 0, 0))
        VV2 = np.stack(VS_transformation.apply_direction(*view_vectors[:, 1, :].T)).T

        VV = np.stack((VV1, VV2), axis=1)
        Pcs_storage = []
        for vec_pair in VV:
            n = np.cross(vec_pair[0], vec_pair[1])
            if np.array_equal(n, np.array([0, 0, 0])):
                continue
            M1 = P1 + np.dot(P2-P1, np.cross(vec_pair[1], n))/np.dot(n, n) * vec_pair[0]
            M2 = P2 + np.dot(P2-P1, np.cross(vec_pair[0], n))/np.dot(n, n) * vec_pair[1]
            Pcs = (M1+M2)/2
            Pcs_storage.append(Pcs)
        SE_transformation = coordinate_systems.get_transformation('Stereo', 'EARTH')
        Pcs_storage = np.array(Pcs_storage)
        Pcs_storage = np.stack(SE_transformation.apply_point(*Pcs_storage.T)).T
        Pcs_storage = np.array([EARTH_frame.toNatural(Pcs) for Pcs in Pcs_storage])
        np.set_printoptions(suppress=True)
        print(Pcs_storage)
    return None