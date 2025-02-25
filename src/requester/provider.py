# provider.py
from enum import Enum
from pystac_client import Client
import planetary_computer


class Provider(Enum):
    PLANETARY_COMPUTER = {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "plot_title": "Planetary Computer",
        "filename": "planetary_computer",
        "qgis_layer_name": "Planetary Computer",
        "collection_mapping": {
            "sentinel-2-l2a": {
                "Red": "B04",
                "Green": "B03",
                "Blue": "B02",
                "True Color": "visual",
                "Near Infrared": "B08",
                "Coastal Aerosol": "B01",
                "Red Edge 1": "B05",
                "Red Edge 2": "B06",
                "Red Edge 3": "B07",
                "Narrow NIR": "B8A",
                "Water Vapor": "B09",
                "SWIR - Cirrus": "B10",
                "SWIR 1": "B11",
                "SWIR 2": "B12",
                "Aerosol Optical Thickness": "AOT",
                "Scene Classification": "SCL",
                "Water Vapor Product": "WVP"
            },
            "landsat-c2-l2": {
                "Red": "red",
                "Green": "green",
                "Blue": "blue",
                "True Color": "red, green, blue",
                "False Color": "nir08, red, green",
                "Near Infrared": "nir08",
                "Coastal Aerosol": "coastal",
                "SWIR 1": "swir16",
                "SWIR 2": "swir22",
                "Thermal Infrared": "lwir11",
                "Aerosol Quality": "qa_aerosol",
                "Pixel QA": "qa_pixel",
                "Radiometric Saturation QA": "qa_radsat",
                "Thermal Atmospheric Transmittance": "atran",
                "Thermal Component Distance": "cdist",
                "Thermal Downwelling Radiance": "drad",
                "Thermal Upwelling Radiance": "urad",
                "Thermal Radiance": "trad",
                "Thermal Emissivity": "emis",
                "Thermal Emissivity Standard Deviation": "emsd",
                "Scene QA": "qa"
            },
            "landsat-c2-l1": {
                "Red": "red",
                "Green": "green",
                "Blue": "blue",
                "True Color": "red, green, blue",
                "False Color": "nir08, red, green",
                "Near Infrared": "nir08",
                "Narrow NIR": "nir09",
                "Pixel QA": "qa_pixel",
                "Radiometric Saturation QA": "qa_radsat"
            }
        }
    }

    EARTH_SEARCH = {
        "url": "https://earth-search.aws.element84.com/v1",
        "plot_title": "Earth Search",
        "filename": "earth_search",
        "qgis_layer_name": "Earth Search",
        "collection_mapping": {
            "landsat-c2-l2": {
                "Red": "red",
                "Green": "green",
                "Blue": "blue",
                "True Color": "red, green, blue",
                "False Color": "nir08, red, green",
                "Near Infrared": "nir08",
                "Coastal Aerosol": "coastal",
                "SWIR 1": "swir16",
                "SWIR 2": "swir22",
                "Thermal Infrared": "lwir11",
                "Aerosol Quality": "qa_aerosol",
                "Pixel QA": "qa_pixel",
                "Radiometric Saturation QA": "qa_radsat",
                "Thermal Atmospheric Transmittance": "atran",
                "Thermal Component Distance": "cdist",
                "Thermal Downwelling Radiance": "drad",
                "Thermal Upwelling Radiance": "urad",
                "Thermal Radiance": "trad",
                "Thermal Emissivity": "emis",
                "Thermal Emissivity Standard Deviation": "emsd",
                "Scene QA": "qa"
            },
            "sentinel-2-l1c": {
                "Red": "red",
                "Green": "green",
                "Blue": "blue",
                "True Color": "visual",
                "Near Infrared": "nir",
                "Coastal Aerosol": "coastal",
                "Red Edge 1": "rededge1",
                "Red Edge 2": "rededge2",
                "Red Edge 3": "rededge3",
                "Narrow NIR": "nir08",
                "Water Vapor": "nir09",
                "SWIR - Cirrus": "cirrus",
                "SWIR 1": "swir16",
                "SWIR 2": "swir22"
            },
            "sentinel-2-l2a": {
                "Red": "red",
                "Green": "green",
                "Blue": "blue",
                "True Color": "visual",
                "Near Infrared": "nir",
                "Coastal Aerosol": "coastal",
                "Red Edge 1": "rededge1",
                "Red Edge 2": "rededge2",
                "Red Edge 3": "rededge3",
                "Narrow NIR": "nir08",
                "Water Vapor": "nir09",
                "SWIR 1": "swir16",
                "SWIR 2": "swir22",
                "Aerosol Optical Thickness": "aot",
                "Scene Classification": "scl",
                "Water Vapor Product": "wvp"
            }
        }
    }

    @property
    def url(self):
        return self.value["url"]

    @property
    def plot_title(self):
        return self.value["plot_title"]

    @property
    def filename(self):
        return self.value["filename"]

    @property
    def qgis_layer_name(self):
        return self.value["qgis_layer_name"]

    @property
    def collection_mapping(self):
        return self.value["collection_mapping"]

    @staticmethod
    def get_client(provider):
        if provider == Provider.PLANETARY_COMPUTER:
            return Client.open(provider.url, modifier=planetary_computer.sign_inplace)
        return Client.open(provider.url)
