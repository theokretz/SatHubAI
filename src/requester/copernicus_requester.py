import rasterio.features
import matplotlib.pyplot as plt
from pystac_client import Client

from .requester import Requester


class CopernicusRequester(Requester):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @staticmethod
    def plot_image(asset_url, title="Copernicus"):
        with rasterio.open(asset_url) as src:
            image = src.read([1, 2, 3])

        # rearranges dimensions
        image = image.transpose((1, 2, 0))

        plt.figure()
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def request_data(self):
        catalog = Client.open("https://catalogue.dataspace.copernicus.eu/stac")

        search = catalog.search(
            collections=["SENTINEL-2"],
            bbox = [16.1826, 48.0996, 16.5771, 48.3235],
            datetime="2023-09-01T00:00:00Z/2023-12-31T23:59:59Z",
        )

        items = search.item_collection()

        selected_item = min(items, key=lambda item: item.properties["cloudCover"])

        asset_url = selected_item.assets['QUICKLOOK'].href
        self.plot_image(asset_url)
