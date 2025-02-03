# stac_requester.py
# credits: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
from .processor.change_detection_processor import ChangeDetectionProcessor
from .processor.processor_factory import ProcessorFactory
from .requester import Requester
from .provider import Provider
from ..utils import display_warning_message, calculate_bbox


class StacRequester(Requester):
    collection_mapping = {
        "Sentinel-2 L1C": "sentinel-2-l1c" ,
        "Sentinel-2 L2A": "sentinel-2-l2a",
        "Landsat Collection 2 L1": "landsat-c2-l1",
        "Landsat Collection 2 L2": "landsat-c2-l2",
    }

    def __init__(self, config, provider, invekos_manager=None):
        super().__init__(config)
        self.config = config
        self.provider = provider
        self.collection = None
        self.invekos_manager = invekos_manager


    def request_data(self):
        """
        Request and process satellite data from a STAC API.

        Returns
        -------
        None
            Processes data through selected processor
        """
        catalog = Provider.get_client(self.provider)

        bbox = calculate_bbox(self.config.coords)

        if self.config.additional_options:
            self.collection =  self.collection_mapping.get(self.config.additional_options.collection)
        else:
            self.collection = "sentinel-2-l2a"

        # Prepare search parameters
        search_params = {
            "collections": self.collection,
            "bbox": bbox,
            "datetime": f"{self.config.start_date}/{self.config.end_date}",
        }

        # add query parameter only if supported - only supported in sentinel-2-l2a collection
        if self.collection == "sentinel-2-l2a":
            search_params["query"] = {
                "eo:cloud_cover": {"lt": 10},
                "s2:nodata_pixel_percentage": {"lt": 1},
            }

        search = catalog.search(**search_params)
        items = search.item_collection()

        if not items:
            display_warning_message("Change your options.", "No satellite data found!")
            return

        # create processor
        processor = ProcessorFactory.create_processor(self.config, self.provider, self.collection, self.invekos_manager)

        # For change detection, use all items
        if isinstance(processor, ChangeDetectionProcessor):
            processor.process(items)
        else:
            # select item with the lowest cloudiness
            selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
            processor.process(selected_item)