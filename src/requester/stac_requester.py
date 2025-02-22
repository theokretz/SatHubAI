# stac_requester.py
# credits: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
import logging

from .processor.change_detection_processor import ChangeDetectionProcessor
from .processor.crop_classification_processor import CropClassificationProcessor
from .processor.processor_factory import ProcessorFactory
from .requester import Requester
from .provider import Provider
from ..utils import display_warning_message, calculate_bbox
import numpy as np

logger = logging.getLogger("SatHubAI.StacRequester")

class StacRequester(Requester):
    collection_mapping = {
        "Sentinel-2 L1C": "sentinel-2-l1c" ,
        "Sentinel-2 L2A": "sentinel-2-l2a",
        "Landsat Collection 2 L1": "landsat-c2-l1",
        "Landsat Collection 2 L2": "landsat-c2-l2",
    }

    def __init__(self, config, provider, invekos_manager=None):
        super().__init__(config)
        self._config = config
        self._provider = provider
        self._collection = None
        self._invekos_manager = invekos_manager


    def request_data(self):
        """
        Request and process satellite data from a STAC API.

        Returns
        -------
        None
            Processes data through selected processor
        """
        catalog = Provider.get_client(self._provider)

        bbox = calculate_bbox(self._config.coords)

        if self._config.additional_options:
            self._collection =  self.collection_mapping.get(self._config.additional_options._collection)
        else:
            self._collection = "sentinel-2-l2a"

        # Prepare search parameters
        search_params = {
            "collections": self._collection,
            "bbox": bbox,
            "datetime": f"{self._config.start_date}/{self._config.end_date}",
        }

        # add query parameter only if supported - only supported in sentinel-2-l2a collection
        if self._collection == "sentinel-2-l2a":
            search_params["query"] = {
                "eo:cloud_cover": {"lt": 10},
                "s2:nodata_pixel_percentage": {"lt": 1},
            }
        else:
            search_params["query"] = {
                "eo:cloud_cover": {"lt": 10},
            }
        print(search_params)
        search = catalog.search(**search_params)
        items = search.item_collection()
        print(items)
        logger.info(f"Found {len(items)} images")

        if not items:
            display_warning_message("Change your options.", "No satellite data found!")
            logger.warning("No satellite data found!")
            return

        # create processor
        processor = ProcessorFactory.create_processor(self._config, self._provider, self._collection, self._invekos_manager)

        # For change detection and crop classification, use all items
        if isinstance(processor, ChangeDetectionProcessor) or isinstance(processor, CropClassificationProcessor):
            processor.process(items)
        else:
            # select item with the lowest cloudiness
            selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
            processor.process(selected_item)


    def request_multiple_images(self, catalog):
        """Needed for TrainingDataProcessor"""
        minx, miny, maxx, maxy = calculate_bbox(self._config.coords)

        # define the max tile size
        TILE_SIZE = 1.0

        lat_splits = np.arange(miny, maxy, TILE_SIZE)
        lon_splits = np.arange(minx, maxx, TILE_SIZE)

        # store all satellite images
        all_items = []
        logger.info(f"Splitting area into {len(lat_splits)} latitude and {len(lon_splits)} longitude steps.")

        # request data for each sub-bbox
        for lat in lat_splits:
            for lon in lon_splits:
                BUFFER = 0.05  # Adjust based on resolution
                sub_bbox = [lon - BUFFER, lat - BUFFER, min(lon + TILE_SIZE + BUFFER, maxx),
                            min(lat + TILE_SIZE + BUFFER, maxy)]

                logger.info(f"Requesting data for sub-BBOX: {sub_bbox}")
                search_params = {
                    "collections": self._collection,
                    "bbox": sub_bbox,
                    "datetime": f"{self._config.start_date}/{self._config.end_date}",
                }
                # add query parameter only if supported - only supported in sentinel-2-l2a collection
                if self._collection == "sentinel-2-l2a":
                    search_params["query"] = {
                        "eo:cloud_cover": {"lt": 10},
                        "s2:nodata_pixel_percentage": {"lt": 1},
                    }

                search = catalog.search(**search_params)
                items = search.item_collection()

                if items:
                    all_items.extend(items)
                    logger.info(f"Found {len(items)} images for BBOX {sub_bbox}")
                    print(f"Found {len(items)} images for BBOX {sub_bbox}")
                else:
                    logger.info(f"Total satellite images collected: {len(all_items)}")

        if not all_items:
            display_warning_message("Expand the area or change options.", "No satellite data found!")
            return None

        return all_items
