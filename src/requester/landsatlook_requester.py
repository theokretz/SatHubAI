

from pystac_client import Client

from .requester import Requester


class LandsatLookRequester(Requester):
    def __init__(self, config):
        super().__init__(config)

    def request_data(self):
        catalog = Client.open("https://landsatlook.usgs.gov/stac-server/")

        search = catalog.search(
            collections=["landsat-c2l2-sr"],
            bbox = [16.1826, 48.0996, 16.5771, 48.3235],
            datetime="2023-01-01T00:00:00Z/2023-12-31T23:59:59Z"
        )

        items = search.item_collection()
        print(items)
        selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
        print(selected_item.assets)