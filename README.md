# QGIS Plugin - SatHubAI

SatHubAI is a QGIS plugin developed as part of my bachelor's thesis to automate satellite data retrieval and agricultural monitoring.

## Features

- Select an area on the map
- Retrieve **true color**, **false color** or **NDVI** images of the selected area for a specific time frame from Sentinel Hub, Planetary Computer and Earth Search
- Optionally, select specific bands
- Download images in the preferred file format and directory
- Import images directly into QGIS
- Plot images
- Calculate NDVI
- Support agricultural monitoring in Austria:
  - **Change Detection**: Identifies abrupt vegetation changes using NDVI.
  - **Crop Classification**: Uses a trained **Random Forest model** to classify fields as monoculture (Reinsaat) or polyculture (Mischsaat).


## Installation

- Install all required Python packages listed in [requirements.txt](./requirements.txt) via the OSGeo4W Shell
  ```
  python -m pip install -r requirements.txt
  ```
- Clone this repository
  ```
  git clone https://github.com/theokretz/SatHubAI.git
  ```
- Copy the plugin folder into your QGIS plugin directory, on Windows usually located at: `C:\Users\User\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`
- Activate the plugin in QGIS:
  - Open QGIS, go to **Plugins > Manage and Install Plugins**
  - Find **SatHubAI** and check the box to activate it

## Configuring Satellite Data Providers
### **Sentinel Hub**
If you want to use data from Sentinel Hub, you need a [Sentinel Hub Account](https://www.sentinel-hub.com/).
Add credentials to: 
```
C:\Users\User\.config\sentinelhub\config.toml
```

Example configuration:
```toml
[default-profile]
instance_id = "instance_id"
sh_client_id = "sh_client_id"
sh_client_secret = "sh_client_secret"
```

### **Earth Search (AWS)**
Some collections (Sentinel-2 L1C, Landsat C2 L2) require AWS credentials. Add them to:

```
C:\Users\User\.aws\credentials
```

Example:

```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

### **Planetary Computer**
No authentication is required.

## Usage

1. **Select Area**: Click and drag to define a area of interest on the QGIS map.
2. **Choose Satellite Provider**: 
   - Sentinel Hub
   - Planetary Computer
   - Earth Search
3. **Select Satellite Collection**: 
   - Sentinel-2 (L1C, L2A)
   - Landsat (C2 L1, C2 L2)
4. **Set Timeframe**: Define the start and end date for image retrieval.
5. **Processing Options**:
   - True/False Color
   - NDVI Calculation
   - Load INVEKOS data
   - Change Detection
   - Crop Classification
6. **Visualization & Export**:
   - Import into QGIS
   - Download as GeoTIFF, PNG, or JPEG
   - Plot images
7. **Submit Request**: The plugin fetches and processes the selected data.

By default, the plugin retrieves **True Color** images from **Sentinel-2 L2A**, unless specified otherwise by the user.


## Acknowledgements
This project uses code from the [Sentinel Hub Python package documentation](https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html) and [Planetary Computer documentation](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/).
