from pathlib import Path
import os
import xarray as xr
import rioxarray as rxr
import cv2 as cv
import numpy as np
import mounttree as mnt
from metpy.calc import pressure_to_height_std, height_to_pressure_std
from metpy.units import units
import scipy.interpolate as sci


def generate_pic_pair_vids(Velox_BT_File, vid_edge_trim):
    """
    Function that generates image pairs from Velox brightness temperature data. The image pairs are each one second
    apart and are saved as grayscale videos in the vids_tmp folder.

    :param Velox_BT_File: Path to Velox NetCDF file containing the brightness temperatures
    :param vid_edge_trim: List specifying how the brightness temperature data set was trimmed in respect to the raw data
    :return: List of paths from image pair videos
    """
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
    """
    Function that searches for identical cloud points in consecutive images. It is based on Shi-Tomasi corner detection
    and calculates the optical flow using the pyramid implication of the Lucas-Kanade algorithm. It also checks the
    quality of the point pairs by a backward calculation of the optical flow, returning only high quality points.

    :param pic_pair_vids_list: List of Paths from image pair videos
    :return: List of arrays containing the image point pairs in pixel units
    """
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
        os.remove(pic_pair_vid)
    return pixel_pairs_list


def viewing_direction(pixel_pairs, Velox_VDC_Data, vid_edge_trim):
    """
    This function translates an array of pixel position pairs into an array of viewing direction vectors. First, it
    calculates how the calibration data set is positioned in relation to the brightness temperature data set. Then a
    decision is made as if the zenith and azimuth angle of a point needs to be interpolated. This is the case if the
    point does not lie directly on a pixel. In the next step, the viewing direction vector is calculated from the zenith
    and azimuth angle found. If there are no four surrounding pixels during the interpolation, a None vector is added.
    This is only the case if the point lies between the center of an edge pixel and the absolute edge of the image and
    the trim of this side is zero. If the translated array contains None vectors, the entire pixel point pair is deleted
    and only valid vectors are returned.

    :param pixel_pairs: Array containing the pixel point pairs of consecutive images
    :param Velox_VDC_Data: Velox calibration file which assigns a viewing direction to each pixel in the camera
                           reference frame
    :param vid_edge_trim: List specifying how the brightness temperature data set was trimmed in respect to the raw data
    :return: Array which contains the translated viewing vectors in the camera reference frame
    """
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
            px_x_coords = z_patch['x-pixel']
            px_y_coords = z_patch['y-pixel']
            z_patch = z_patch.values*np.pi/180
            a_patch = a_patch.values*np.pi/180
            # circularity consideration of the zenith angle data not necessary since the breakpoint is behind the camera
            z = sci.RegularGridInterpolator((px_x_coords, px_y_coords), z_patch)(pixel).item()
            # circularity consideration of the azimuth angle data
            a_sin = sci.RegularGridInterpolator((px_x_coords, px_y_coords), np.sin(a_patch))(pixel)
            a_cos = sci.RegularGridInterpolator((px_x_coords, px_y_coords), np.cos(a_patch))(pixel)
            a = np.arctan2(a_sin, a_cos).item()
            VD_Vector = (np.tan(z)*np.cos(a), -np.tan(z)*np.sin(a), 1)
            VD_Vector_storage.append(VD_Vector)
    VD_Vector_array = np.array(VD_Vector_storage).reshape(-1,2,3)
    VD_Vector_clean = (np.delete(VD_Vector_array, np.unique(np.where(VD_Vector_array == None)[0]), axis=0)
                       .astype(np.float64))
    return VD_Vector_clean


def cloud_point_filter(Pcs_storageN, ref_height, DSM_Data):
    """
    This function filters calculated cloud points according to plausibility. To do this, the heights are checked. Cloud
    points cannot be higher than the stereo reference system, as Velox looks in the nadir direction and does not see
    above the horizon even when HALO is inclined. The height of the reference system corresponds approximately to the
    average height of HALO during the recording of the image pair. Furthermore, no cloud points are used that are below
    the height of the 1000 hPa layer according to the U.S. standard atmosphere. In reality, such clouds do exist, but
    there is no ERA5 wind data available for them, which is essential for wind correction. The points are then checked
    again using a digital surface model. This ensures that no points lie below the ground or that the earth's surface is
    incorrectly tracked. Points found less than one hundred meters above the surface are returned. If no or a spatially
    incomplete surface model is passed to the function, sea level is assumed at the undefined positions, whereby the
    lowest points are removed by the 1000 hPa filter.

    :param Pcs_storageN: Array of cloud points in natural coordinates of WGS84
    :param ref_height: Height of the stereo reference system above WGS84
    :param DSM_Data: Data of the digital surface model
    :return: Array of indices for cloud points that need to be filtered out
    """
    Pcs_index = []
    height_1000hPa = pressure_to_height_std(1000*units.hPa).to_base_units().magnitude
    DSM_exists = False
    if np.shape(DSM_Data) != ():
        DSM_exists = True
        lat_coords = DSM_Data['y'].values
        lon_coords = DSM_Data['x'].values
    for Pi, Pcs in enumerate(Pcs_storageN):
        if Pcs[2] > ref_height:
            Pcs_index.append(Pi)
            continue
        if Pcs[2] < height_1000hPa:
            Pcs_index.append(Pi)
            continue
        if DSM_exists:
            if (np.min(lat_coords) <= Pcs[0] <= np.max(lat_coords) and
                    np.min(lon_coords) <= Pcs[1] <= np.max(lon_coords)):
                Pcs_lat_index = np.argmin(np.abs(lat_coords - Pcs[0]))
                Pcs_lon_index = np.argmin(np.abs(lon_coords - Pcs[1]))
                surface_height = DSM_Data[0, Pcs_lat_index, Pcs_lon_index].values
                if Pcs[2] < surface_height + 100:
                    Pcs_index.append(Pi)
    return np.array(Pcs_index)


def stereographic_reconstruction(pic_pair_vids_list, pixel_pairs_list, Velox_VDC_File, HALO_IRS_File, MNT_File,
                                 vid_edge_trim, ERA5_UV_Wind_File, DSM_file):
    """
    This function performs a stereographic reconstruction of cloud points based on the method presented by Kölling
    et al. (2019) and corrects the wind-cloud shift according to the technique of Volkmer et al. (2024).

    :param pic_pair_vids_list: List of Paths from image pair videos
    :param pixel_pairs_list: List of Arrays containing the image point pairs in pixel units
    :param Velox_VDC_File: Velox calibration file which assigns a viewing direction to each pixel in the camera
                           reference frame
    :param HALO_IRS_File: HALO position file
    :param MNT_File: File describing the used coordinate systems and their relation to each other
    :param vid_edge_trim: List specifying how the brightness temperature data set was trimmed in respect to the raw data
    :param ERA5_UV_Wind_File: Wind data file
    :param DSM_file: Digital surface model file
    :return: Nested list containing the times of reconstruction [0], the reference point positions of the stereo
             coordinate system [1] and the arrays of reconstructed cloud points [2]
    """
    # load the data files
    coordinate_systems = mnt.load_mounttree(MNT_File)
    Velox_VDC_Data = xr.open_dataset(Velox_VDC_File)
    HALO_IRS_Data = xr.open_dataset(HALO_IRS_File)
    IRS_TIME = HALO_IRS_Data['time']   # time
    IRS_LAT = HALO_IRS_Data['IRS_LAT'] # Latitude
    IRS_LON = HALO_IRS_Data['IRS_LON'] # Longitude
    IRS_ALT = HALO_IRS_Data['IRS_ALT'] # Altitude
    IRS_PHI = HALO_IRS_Data['IRS_PHI'] # Roll
    IRS_THE = HALO_IRS_Data['IRS_THE'] # Pitch
    IRS_HDG = HALO_IRS_Data['IRS_HDG'] # Yaw
    UV_Wind_Data = xr.open_dataset(ERA5_UV_Wind_File)
    UV_Wind_time = UV_Wind_Data['valid_time']
    UV_Wind_lat = UV_Wind_Data['latitude']
    UV_Wind_lon = UV_Wind_Data['longitude']
    UV_Wind_pl = UV_Wind_Data['pressure_level']
    U_Wind = UV_Wind_Data['u']
    V_Wind = UV_Wind_Data['v']
    # check if DSM_file is provided
    if DSM_file != '':
        DSM_Data = rxr.open_rasterio(DSM_file)
    else:
        DSM_Data = None

    cloud_point_main_storage = [[],[],[]]
    # perform reconstruction for every video
    for vid in range(len(pic_pair_vids_list)):
        vid_time = pic_pair_vids_list[vid][-33:-4]
        time_index = np.where(IRS_TIME.astype(str) == vid_time)[0].item()
        # reconstruction doesn't work if P1=P2
        if (IRS_LAT[time_index] == IRS_LAT[time_index+1] and IRS_LON[time_index] == IRS_LON[time_index+1] and
                IRS_ALT[time_index] == IRS_ALT[time_index+1]):
            continue
        # calculate stereo reference system
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
        PrefN = EARTH_frame.toNatural(PrefE)
        ref_lat, ref_lon, ref_height = PrefN

        # test if pixel pairs exist
        if np.shape(pixel_pairs_list[vid]) == ():
            continue
        # calculate view vectors of pixel pairs
        view_vectors = viewing_direction(pixel_pairs_list[vid], Velox_VDC_Data, vid_edge_trim)
        # test if usable view vectors have been calculated
        if view_vectors.shape == (0, 2, 3):
            continue

        # calculation of the aircraft positions and viewing directions in the stereo reference system
        coordinate_systems.update(lat=IRS_LAT[time_index], lon=IRS_LON[time_index], height=IRS_ALT[time_index],
                                  roll=IRS_PHI[time_index], pitch=IRS_THE[time_index], yaw=IRS_HDG[time_index],
                                  ref_lat=ref_lat, ref_lon=ref_lon, ref_height=ref_height)
        VS_transformation = coordinate_systems.get_transformation('VELOX', 'STEREO')
        P1 = np.array(VS_transformation.apply_point(0, 0, 0))
        VV1 = np.stack(VS_transformation.apply_direction(*view_vectors[:,0,:].T)).T

        coordinate_systems.update(lat=IRS_LAT[time_index+1], lon=IRS_LON[time_index+1], height=IRS_ALT[time_index+1],
                                  roll=IRS_PHI[time_index+1], pitch=IRS_THE[time_index+1], yaw=IRS_HDG[time_index+1])
        VS_transformation = coordinate_systems.get_transformation('VELOX', 'STEREO')
        P2 = np.array(VS_transformation.apply_point(0, 0, 0))
        VV2 = np.stack(VS_transformation.apply_direction(*view_vectors[:, 1, :].T)).T

        # first point reconstruction
        VV = np.stack((VV1, VV2), axis=1)
        Pcs_storage = []
        VV_storage = []
        for vec_pair in VV:
            n = np.cross(vec_pair[0], vec_pair[1])
            # test that view vectors from P1 and P2 are not identical for the same tracked point
            if np.array_equal(n, np.array([0, 0, 0])):
                continue
            # calculation of the mis-pointing vector and cloud point
            M1 = P1 + np.dot(P2-P1, np.cross(vec_pair[1], n))/np.dot(n, n) * vec_pair[0]
            M2 = P2 + np.dot(P2-P1, np.cross(vec_pair[0], n))/np.dot(n, n) * vec_pair[1]
            Pcs = (M1+M2)/2
            Pcs_storage.append(Pcs)
            VV_storage.append(vec_pair)
        VV = np.array(VV_storage)
        if len(Pcs_storage) == 0:
            continue

        # conversion of cloud points from the stereo reference system (S) to the Cartesian earth system (E) and then to
        # natural coordinates of the earth system (N)
        SE_transformation = coordinate_systems.get_transformation('STEREO', 'EARTH')
        Pcs_storageS = np.array(Pcs_storage)
        Pcs_storageE = np.stack(SE_transformation.apply_point(*Pcs_storageS.T)).T
        Pcs_storageN = np.array([EARTH_frame.toNatural(Pcs) for Pcs in Pcs_storageE])
        # filtering of the cloud points for plausibility
        filter_indices = cloud_point_filter(Pcs_storageN, ref_height, DSM_Data)
        if np.shape(filter_indices) != (0,):
            Pcs_storageN = np.delete(Pcs_storageN, filter_indices, axis=0)
            Pcs_storageE = np.delete(Pcs_storageE, filter_indices, axis=0)
            VV = np.delete(VV, filter_indices, axis=0)
            # test if there are still some cloud points after filtering
            if np.shape(Pcs_storageN) == (0, 3):
                continue

        # calculation of the time index in the ERA5 wind data and the position of the image pair in the time interval
        time_index = np.where(UV_Wind_time.astype(str) == vid_time[:-15]+'00:00.000000000')[0].item()
        time_position = int(vid_time[-15:-13])*60+float(vid_time[-12:])+0.5
        # iterative wind correction
        number_of_iterations = 5
        for iteration in range(number_of_iterations):
            PcsW_storage = []
            for PcsN, PcsE, vec_pair in zip(Pcs_storageN, Pcs_storageE, VV):
                # indices of the surrounding wind data points
                lat_index = np.where(UV_Wind_lat == np.floor(PcsN[0]*4)/4)[0].item()
                lon_index = np.where(UV_Wind_lon == np.floor(PcsN[1]*4)/4)[0].item()
                pl_height = height_to_pressure_std(PcsN[2]*units.meter).magnitude
                pl_index = np.argmin(np.abs(UV_Wind_pl.values - pl_height))
                if UV_Wind_pl[pl_index] < pl_height:
                    pl_index -= 1

                # selection of the wind field surrounding the cloud point
                # Three height layers are used, as with only two a cloud point just below a height layer can lie outside
                # the convex hull when converted to Cartesian coordinates due to the curvature of the earth.
                if np.floor(PcsN[1]*4)/4 == 179.75:
                    # in the case you are flying close to the antimeridian
                    lon180 = np.where(UV_Wind_lon == -180.)[0].item()
                    u_wind_patch = xr.concat((U_Wind[time_index:time_index+2, pl_index:pl_index+3,
                                             lat_index-1:lat_index+1, lon_index:lon_index+1],
                                             U_Wind[time_index:time_index+2, pl_index:pl_index+3,
                                             lat_index-1:lat_index+1, lon180:lon180+1]), dim='longitude')
                    v_wind_patch = xr.concat((V_Wind[time_index:time_index+2, pl_index:pl_index+3,
                                             lat_index-1:lat_index+1, lon_index:lon_index+1],
                                             V_Wind[time_index:time_index+2, pl_index:pl_index+3,
                                             lat_index-1:lat_index+1, lon180:lon180+1]), dim='longitude')
                else:
                    u_wind_patch = U_Wind[time_index:time_index+2, pl_index:pl_index+3,
                                   lat_index-1:lat_index+1, lon_index:lon_index+2]
                    v_wind_patch = V_Wind[time_index:time_index+2, pl_index:pl_index+3,
                                   lat_index-1:lat_index+1, lon_index:lon_index+2]

                # generating the interpolation grid points in the required Cartesian coordinates
                alt_height = (pressure_to_height_std(u_wind_patch['pressure_level'].values*units.hPa)
                              .to_base_units().magnitude)
                patch_coords = np.array(np.meshgrid(u_wind_patch['latitude'], u_wind_patch['longitude'],
                                                    alt_height)).T.reshape(-1, 3)

                patch_coordsE = np.array([EARTH_frame.toCartesian(GP) for GP in patch_coords])
                grid_points = np.concatenate((np.c_[patch_coordsE, [0]*12], np.c_[patch_coordsE, [3600]*12]))

                # interpolation of the wind vector
                u_wind = sci.griddata(grid_points, u_wind_patch.values.ravel(), np.append(PcsE, time_position))
                v_wind = sci.griddata(grid_points, v_wind_patch.values.ravel(), np.append(PcsE, time_position))
                wind_vector = np.array([v_wind.item(), u_wind.item(), 0])

                # point reconstruction with wind correction
                P1W = P1 + wind_vector/2
                P2W = P2 - wind_vector/2
                n = np.cross(vec_pair[0], vec_pair[1])
                M1W = P1W + np.dot(P2W-P1W, np.cross(vec_pair[1], n))/np.dot(n, n) * vec_pair[0]
                M2W = P2W + np.dot(P2W-P1W, np.cross(vec_pair[0], n))/np.dot(n, n) * vec_pair[1]
                PcsW = (M1W+M2W)/2
                PcsW_storage.append(PcsW)

            # conversion of the wind-corrected points from the stereo system to the Cartesian and natural earth system
            Pcs_storageS = np.array(PcsW_storage)
            Pcs_storageE = np.stack(SE_transformation.apply_point(*Pcs_storageS.T)).T
            Pcs_storageN = np.array([EARTH_frame.toNatural(Pcs) for Pcs in Pcs_storageE])

            # filter and delete dubiously corrected points
            if iteration+1 == number_of_iterations:
                filter_indices = cloud_point_filter(Pcs_storageN, ref_height, DSM_Data)
            else:
                filter_indices = cloud_point_filter(Pcs_storageN, ref_height, DSM_Data=None)
            if np.shape(filter_indices) != (0,):
                Pcs_storageN = np.delete(Pcs_storageN, filter_indices, axis=0)
                Pcs_storageE = np.delete(Pcs_storageE, filter_indices, axis=0)
                VV = np.delete(VV, filter_indices, axis=0)
                if np.shape(Pcs_storageN) == (0, 3):
                    break
        # check that there are reconstructed cloud points
        if np.shape(Pcs_storageN) == (0, 3):
            continue
        # saving data to main storage
        cloud_point_main_storage[0].append(np.datetime64(vid_time)+np.timedelta64(500, 'ms'))
        cloud_point_main_storage[1].append(PrefN)
        cloud_point_main_storage[2].append(Pcs_storageN)
    return cloud_point_main_storage


def save_to_NetCDF(cloud_point_main_storage, Pcs_save_path):
    """
    This function saves the reconstructed cloud points and the position of the corresponding stereo reference point in a
    NetCDF file. Since the number of reconstructed points per image pair can vary, the individual cloud point arrays are
    evenly filled with np.nan values to obtain the same shape.

    :param cloud_point_main_storage: Nested list containing the times of reconstruction [0], the reference point
                                     positions of the stereo coordinate system [1] and the arrays of reconstructed cloud
                                     points [2]
    :param Pcs_save_path: Path to the location where the cloud points are stored in the form of a NetCDF file
    :return: Boolean whether cloud points could be saved
    """
    if len(cloud_point_main_storage[0]) == 0:
        return False
    Nr = [len(Pcs) for Pcs in cloud_point_main_storage[2]]
    Nr_max = np.max(Nr)
    cloud_point_main_storage[2] = [np.concatenate((Pcs, np.array([np.nan]*3*(Nr_max-Nr[Pi])).reshape(-1, 3)))
                                   for Pi, Pcs in enumerate(cloud_point_main_storage[2])]

    data_vars = {}
    data_vars['ref_lat'] = (('time'), [PrefN[0] for PrefN in cloud_point_main_storage[1]])
    data_vars['ref_lon'] = (('time'), [PrefN[1] for PrefN in cloud_point_main_storage[1]])
    data_vars['ref_alt'] = (('time'), [PrefN[2] for PrefN in cloud_point_main_storage[1]])
    data_vars['Pcs_lat'] = (('time', 'Nr'), [Pcs[:, 0] for Pcs in cloud_point_main_storage[2]])
    data_vars['Pcs_lon'] = (('time', 'Nr'), [Pcs[:, 1] for Pcs in cloud_point_main_storage[2]])
    data_vars['Pcs_alt'] = (('time', 'Nr'), [Pcs[:, 2] for Pcs in cloud_point_main_storage[2]])

    DataNetCDF = xr.Dataset(data_vars=data_vars,
                            coords={'time': cloud_point_main_storage[0],
                                    'Nr': range(1, Nr_max+1)},
                            attrs={'description': 'Cloud points reconstructed by Silver'})
    DataNetCDF['time'].attrs = {'description': 'Mean recording time of the image pair used for reconstruction'}
    DataNetCDF['Nr'].attrs = {'description': 'Key number of the cloud point'}
    DataNetCDF['ref_lat'].attrs = {'description': 'Latitude of the stereo reference point', 'units': 'Degree',
                                   'reference system': 'WGS84'}
    DataNetCDF['ref_lon'].attrs = {'description': 'Longitude of the stereo reference point', 'units': 'Degree',
                                   'reference system': 'WGS84'}
    DataNetCDF['ref_alt'].attrs = {'description': 'Altitude of the stereo reference point', 'units': 'Meter',
                                   'reference system': 'WGS84'}
    DataNetCDF['Pcs_lat'].attrs = {'description': 'Latitude of the cloud point', 'units': 'Degree',
                                   'reference system': 'WGS84'}
    DataNetCDF['Pcs_lon'].attrs = {'description': 'Longitude of the cloud point', 'units': 'Degree',
                                   'reference system': 'WGS84'}
    DataNetCDF['Pcs_alt'].attrs = {'description': 'Altitude of the cloud point', 'units': 'Meter',
                                   'reference system': 'WGS84'}

    DataNetCDF.to_netcdf(path=Pcs_save_path)
    return True