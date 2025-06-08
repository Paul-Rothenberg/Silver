<h2 align="center">
<img src="https://github.com/Paul-Rothenberg/Silver/blob/main/Logo.png" width="400"><br>
Stereographic Infrared reconstruction of<br>cLoud points with the airborne VELOX imagER
</h2>

# Silver
Silver is a software for data processing. It enables the evaluation of brightness temperature datasets recorded with an airborne VELOX imager (VELOX327k veL). Via the technique of stereography, cloud surface points can be derived from this data. The program was developed specifically for the imager on board the High Altitude and LOng Range Research Aircraft (HALO) and is adapted to it. However, reconfiguration to other platforms is possible.

**Important: The camera calibration is still in an early state, limiting Silver's operational capability at present. Please check Silver-Thesis.pdf to ensure it suits your requirements or wait for upcoming releases!**

#
## Installation guide:
1. Clone the repository
2. Install dependencies
3. Run silver.py (will process the example dataset)
4. If processing is complete and the file Pcs_test.nc has been generated in the Sample folder, the installation was successful
5. You are now free to adapt namelist.py, velox.yaml or other program parts to the requirements of your own reconstruction

## File explanation
- **silver.py** Contains the main program
- **silver_lib.py** Provides the set of all functions required for reconstruction
- **namelist.py** Configuration file
- **velox.yaml** Defines utilized coordinate systems
- **Velox-VDC.nc** Viewing direction calibration of the VELOX imager (Early state: Not recommended for general scientific evaluation!)

## Dependencies:
pathlib, os, multiprocessing, [xarray](https://anaconda.org/conda-forge/xarray), [rioxarray](https://anaconda.org/conda-forge/rioxarray), [opencv](https://anaconda.org/anaconda/opencv), [numpy](https://anaconda.org/conda-forge/numpy), [mounttree](https://pypi.org/project/mounttree), [metpy](https://anaconda.org/conda-forge/metpy), [scipy](https://anaconda.org/conda-forge/scipy)

## Namelist Parameter
- **Velox_BT_File** Path to VELOX brightness temperature dataset (Halo-(AC)³ Format)
- **Velox_VDC_File** Path to Velox-VDC.nc
- **HALO_IRS_File** Path to aircraft position data from BAHAMAS
- **MNT_File** Path to velox.yaml
- **vid_edge_trim** Number of pixels missing at each edge in the dataset in relation to the raw data - [Bow, Starboard, Stern, Port]
- **ERA5_UV_Wind_File** Path to ERA5 wind field
- **DSM_file** Path to digital surface model
- **Pcs_save_path** Save location for reconstructed cloud points
- **process_number** Maximum number of parallel running reconstructions
- **low_RAM** Set to true if datasets should not be loaded into RAM completely (resource-saving at the expense of slower processing)

---
<h2 align="center">For a very detailed explanation and documentation, check out Silver-Thesis.pdf</h2>
