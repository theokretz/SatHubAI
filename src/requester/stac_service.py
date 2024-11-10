# stac_service.py
from enum import Enum
from pystac_client import Client
import planetary_computer

class StacService:
    class Provider(Enum):
        PLANETARY_COMPUTER = (
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            "Planetary Computer Image",
            "planetary_computer_image",
            "Planetary Computer Layer"
        )
        EARTH_SEARCH = (
            "https://earth-search.aws.element84.com/v1",
            "Earth Search Image",
            "earth_search_image",
            "Earth Search Layer"
        )

        @property
        def url(self):
            return self.value[0]

        @property
        def plot_title(self):
            return self.value[1]

        @property
        def filename(self):
            return self.value[2]

        @property
        def qgis_layer_name(self):
            return self.value[3]


    @staticmethod
    def get_client(provider):
        if provider == StacService.Provider.PLANETARY_COMPUTER:
            return Client.open(provider.url, modifier=planetary_computer.sign_inplace)
        return Client.open(provider.url)
