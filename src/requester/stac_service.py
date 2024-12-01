# stac_service.py
from enum import Enum
from pystac_client import Client
import planetary_computer

class StacService:
    class Provider(Enum):
        PLANETARY_COMPUTER = {
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "plot_title": "Planetary Computer",
            "filename": "planetary_computer",
            "qgis_layer_name": "Planetary Computer",
            "collection_mapping": {
                "sentinel-2-l2a" : {
                    "Red": "B04",
                    "Green": "B03",
                    "Blue": "B02",
                    "True Color": "visual",
                    "Near Infrared": "B08"
                },
                "landsat-c2-l2" : {
                    "Red": "red",
                    "Green": "green",
                    "Blue": "blue",
                    "Near Infrared": "nir08"
                },
                "landsat-c2-l1" : {
                    "Red": "red",
                    "Green": "green",
                    "Near Infrared": "nir08"
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
                    "Near Infrared": "nir08"
                },
                "sentinel-2-l1c": {
                    "Red": "red",
                    "Green": "green",
                    "Blue": "blue",
                    "Near Infrared": "nir"
                },
                "sentinel-2-l2a" : {
                    "Red": "red",
                    "Green": "green",
                    "Blue": "blue",
                    "True Color": "visual",
                    "Near Infrared": "nir"
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
        if provider == StacService.Provider.PLANETARY_COMPUTER:
            return Client.open(provider.url, modifier=planetary_computer.sign_inplace)
        return Client.open(provider.url)
