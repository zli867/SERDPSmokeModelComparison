# ModelComparisons

ModelComparisons is a collection of Python scripts and utility functions developed to compare fire-related modelling systems such as WRF-SFIRE, CMAQ, and BlueSky. It contains routines for extracting emissions, plume heights, and $PM_{2.5}$ concentrations from model outputs, performing cross-model comparisons, and generating publication-quality figures based on the results.

## Project Structure

The repository is organised into sub-packages according to the type of analysis performed. The following table summarises each directory:

| Directory             | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| Emissions/            | Extraction and comparison of emissions from BlueSky output and WRF-SFIRE simulations. Includes routines to compute hourly emission fractions for flaming, smouldering, and residual phases, and scripts to aggregate daily totals. |
| PlumeHeight/          | Tools for reading plume height outputs from WRF-SFIRE and CMAQ and comparing different plume rise schemes. Timing analyses of plume height are also included. |
| Concentration/        | Scripts for extracting $PM_{2.5}$ concentrations from CMAQ and WRF-SFIRE NetCDF files, mapping concentrations to monitoring locations, and comparing across plume schemes. |
| StatisticalAnalysis/  | Loads aggregated statistics (burned area, consumption, emissions, plume height) from stats_data and computes summary measures used in the associated study. |
| MonitorInfo/          | Metadata and plotting styles for air-quality monitors used in the analyses. |
| WindEvaluation/       | Functions for extracting wind components from WRF output and converting between wind speed/direction and u/v components. |
| TimingSensitivity/    | Evaluates the sensitivity of modelled concentrations to ignition-time differences between WRF-SFIRE and CMAQ simulations. |
| FRPSensitivity/       | Assesses how variations in fire radiative power (FRP) parameters impact plume height and concentration. |
| UncertainWindAnalysis/| Routines for quantifying the effect of wind-field uncertainty on smoke dispersion. Contains utilities for trajectory calculation, least-squares assimilation, and other advanced analyses. |
| PublicationFigures/   | Collection of scripts used to generate the figures for the companion publication. These call functions from other modules and produce plots saved to disk. |
| static/               | Static data used by the scripts, including bsp_fuel_type.csv (mapping BlueSky fuel categories), fccs2_fuelload.nc (fuel load look-up table), and namelist.fire_emissions_RADM2GOCART (WRF-SFIRE configuration). |
| stats_data/           | Pickled Python objects (*.pickle) containing aggregated statistics such as burned area, consumption, emissions, plume height, and concentration. These are used by the StatisticalAnalysis and PublicationFigures modules. |
| utils.py              | Shared utility functions for reading WRF and CMAQ grid information, converting coordinates, creating discrete colour maps, and computing time arrays. |

## Requirements

The code is tested under Python 3.8–3.11. It relies on a number of scientific Python packages and GIS libraries:

- **NumPy** and **SciPy** for array operations and statistics
- **Pandas** for data handling
- **Matplotlib** for plotting
- **NetCDF4** for reading NetCDF files
- **Statsmodels** for regression analyses
- **Scikit-learn** for linear regression
- **PyProj** for coordinate transformations
- **Shapely** and **Fiona** for geometric operations and reading shapefiles

Installing Shapely and Fiona can be challenging on some systems because they depend on GEOS and GDAL. It is recommended to use a conda environment, which will install these compiled dependencies automatically.

## Data Preparation

Most scripts in this repository assume that you have model output and observational data stored locally. Before running a script, open it in a text editor and modify the file paths to point to your own data.

If you are using different directories for your data, search for variables such as `conc_filename`, `wrf_filename`, `cmaq_filename`, or similar in the scripts and edit them accordingly.

## Data Structure from Field Campaigns
### Pollutant Concentration Data
#### Data format:
Processed the observation data and generate a hourly observation data in csv format. 

| UTC_time | monitor |lon|lat|pollutant<sub>1</sub>|pollutant<sub>2</sub>|...|pollutant<sub>n</sub>|
| ------ | ----| ------ | ----| ------ | ------ |------ |------ |
| YYYY-MM-DD HH:MM:SS| monitor name| monitor longitude location | monitor latitude location| concentration |concentration |...|concentration|
#### Data units:
* Longitude: degree
* Latitude: degree
* PM25: $\mu g/ m^3$
* BC: $\mu g/ m^3$
* O3: ppb
* CO: ppb

### Wind Observation Data
#### Data format:
Processed the observation data and generate a hourly observation data in csv format. 

| UTC_time | monitor |lon|lat|wdspd|wddir|elevation|
| ------ | ----| ------ | ----| ------ | ------ | ------ |
| YYYY-MM-DD HH:MM:SS| monitor name| monitor longitude location | monitor latitude location|wind speed|wind direction|monitor elevation|
#### Data units:
* Longitude: degree
* Latitude: degree
* Wind speed: m/s
* Wind direction: degree
* Elevation: m


### Fire Data
#### Data format:
Processed the reported prescribed burning information to json data.
```
{
  "fires": [
    {
      "id": fire_1_name,
      "date": "YYYY-MM-DD",
      "start_UTC": "YYYY-MM-DD HH:MM:SS"",
      "end_UTC": "YYYY-MM-DD HH:MM:SS",
      "lat": centroid_latitude_of_burned_region,
      "lng": centroid_longitude_of_burned_region,
      "burned_area": area_values,
      "type": burn_type,
      "perimeter": burn_area_polygon,
      "ignition_patterns": [
        {
          "ignition_time": [],
          "ignition_lat": [],
          "ignition_lng": []
        }
      ]
    },
    ...
    {
      "id": fire_n_name,
      "date": "YYYY-MM-DD",
      "start_UTC": "YYYY-MM-DD HH:MM:SS",
      "end_UTC": "YYYY-MM-DD HH:MM:SS",
      "lat": centroid_latitude_of_burned_region,
      "lng": centroid_longitude_of_burned_region,
      "burned_area": area_values,
      "type": burn_type,
      "perimeter": burn_area_polygon,
      "ignition_patterns": [
        {
          "ignition_time": [YYYY-MM-DD HH:MM:SS],
          "ignition_lat": [],
          "ignition_lng": []
        }
      ]
    }
  ]
}
```
### Variables and data units: 
* id: FtBn + Burn Units
* date: UTC date
* start_UTC: start UTC time of fire
* end_UTC: end UTC time of fire
* lat: centroid latitude of burned region, units: degree
* lng: centroid longitude of burned region, units: degree
* burned_area: burned area of fires, units: acres
* type: rx for prescribed burning; wf for wildfires
* perimeter: boundary of burned area polygons
* ignition_patters: ignition lines for prescribed burnings. It is a list and each elements in the list denotes an ignition line. The ignition line defined by the ignition time, coordinates of ignition lines.
  * ignition_time:  UTC time, YYYY-MM-DD HH:MM:SS
  * ignition_lat: units: degree
  * ignition_lng: units: degree


## Using the Scripts

This repository is not packaged as a single executable; instead, each script is intended to be run interactively or via the Python interpreter after setting appropriate file paths.

## Generating Publication Figures

To reproduce the figures used in the associated publication, run the scripts in the `PublicationFigures/` folder. Each script reads model outputs, performs the necessary analysis, and saves a figure, typically as a PNG or PDF. Ensure that all file paths are updated to reflect your data. For example:

```bash
python PublicationFigures/ModelPerformance.py
```

This will generate several plots illustrating model performance metrics.


## Citation

If you use this code in a publication, please cite the accompanying paper (once available) and acknowledge the authors. You may also cite this repository directly.

---

This README provides a high-level overview of the scripts and data in ModelComparisons. For detailed usage, open the individual files and refer to the docstrings and comments within each script. Because the code depends on external model outputs and observation data, it is essential to customise file paths before running the scripts. If you encounter issues or have suggestions for improvement, please reach out or submit a pull request.