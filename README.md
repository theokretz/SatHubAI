# QGIS Plugin - SatHubAI

This PyQGIS plugin was developed as part of my bachelor's thesis.

## Current Features

- Lets the User select an area on the map
- Returns a True Color Image of the selected area for a specific time frame using data from Sentinel Hub/Planetary Computer/Earth Search
- Downloads the True Color Image in preferred file format and preferred directory
- Calculates NDVI

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

## Configuring Sentinel Hub
If you want to use data from Sentinel Hub, you need a [Sentinel Hub Account](https://www.sentinel-hub.com/).
-  Add credentials to `C:\Users\User\.config\sentinelhub\config.toml`

```toml
[default-profile]
instance_id = "instance_id"
sh_client_id = "sh_client_id"
sh_client_secret = "sh_client_secret"
```

## Acknowledgements
This project uses code from the [Sentinel Hub Python package documentation](https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html) and [Planetary Computer documentation](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/).
