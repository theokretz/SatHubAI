# sentinel_hub_requester.py
# credits: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from qgis._core import QgsProject, QgsMessageLog, Qgis
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
)
import tempfile
from rasterio.transform import from_bounds

from ..exceptions.missing_credentials_exception import MissingCredentialsException
from .requester import Requester
from ..utils import import_into_qgis, display_error_message


class SentinelHubRequester(Requester):
    def __init__(self, config):
        super().__init__(config)
        self.sh_config = SHConfig()
        self.config = config

        if not self.sh_config.sh_client_id or not self.sh_config.sh_client_secret:
            raise MissingCredentialsException("Error! To use Process API, please provide the credentials (OAuth client ID and client secret).")

        self.resolution_mapping = {
             DataCollection.SENTINEL2_L1C : 10,
             DataCollection.SENTINEL2_L2A : 10,
             DataCollection.LANDSAT_MSS_L1 : 60,
             DataCollection.LANDSAT_TM_L1 : 30,
             DataCollection.LANDSAT_TM_L2 : 30,
             DataCollection.LANDSAT_ETM_L1 : 30,
             DataCollection.LANDSAT_ETM_L2 : 30,
             DataCollection.LANDSAT_OT_L1 : 30,
             DataCollection.LANDSAT_OT_L2 : 30,
        }

        self.mime_type_mapping = {
            "TIFF": MimeType.TIFF,
            "PNG": MimeType.PNG,
            "JPEG": MimeType.JPG
        }

        self.collection_mapping = {
            "Sentinel-2 L1C": DataCollection.SENTINEL2_L1C,
            "Sentinel-2 L2A": DataCollection.SENTINEL2_L2A,
            "Landsat 1-5 MSS L1": DataCollection.LANDSAT_MSS_L1,
            "Landsat 4-5 TM L1": DataCollection.LANDSAT_TM_L1,
            "Landsat 4-5 TM L2": DataCollection.LANDSAT_TM_L2,
            "Landsat 7 ETM+ L1": DataCollection.LANDSAT_ETM_L1,
            "Landsat 7 ETM+ L2": DataCollection.LANDSAT_ETM_L2,
            "Landsat 8-9 OLI/TIRS L1": DataCollection.LANDSAT_OT_L1,
            "Landsat 8-9 OLI/TIRS L2": DataCollection.LANDSAT_OT_L2,
        }

    @staticmethod
    def plot_image(image, factor=1.0, clip_range=None, normalize=True):
        if normalize:
            image = (image - image.min()) / (image.max() - image.min())
        image *= factor
        if clip_range:
            image = np.clip(image, clip_range[0], clip_range[1])
        plt.figure()
        plt.imshow(image)
        plt.title("Sentinel Hub")
        plt.axis('off')
        plt.show()

    @staticmethod
    def import_into_qgis_without_download(image, size, bbox):
        """load a satellite image into QGIS without downloading it first via temporary file."""
        lower_left = bbox.lower_left
        upper_right = bbox.upper_right

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            with rasterio.open(
                    tmp_file.name, 'w',
                    driver='GTiff',
                    width=size[0],
                    height=size[1],
                    count=3,
                    dtype=image.dtype,
                    transform=from_bounds(lower_left[0], lower_left[1], upper_right[0], upper_right[1],
                    width=size[0],
                    height=size[1]),
                    crs="EPSG:4326"
            ) as tmp_dataset:
                tmp_dataset.write(image.transpose(2, 0, 1))

            import_into_qgis(tmp_file.name, "Sentinel Hub Layer")

    @staticmethod
    def get_evalscript_for_collection(collection):
        """generates an evalscript based on the satellite data collection."""

        collection_name = collection.api_id
        print(collection_name)

        if "sentinel-2" in collection_name:
            return """
            //VERSION=3
            function setup(){
              return{
                input: ["B02", "B03", "B04"], 
                output: {bands: 3}
              }
            }
            
            function evaluatePixel(sample){
              return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5];
            }
            """
        elif "landsat" in collection_name:
            return """
                //VERSION=3
                function setup() {
                    return {
                        input: [{ 
                            bands: ["B01", "B02", "B03"] 
                        }],
                        output: { 
                            bands: 3 
                        }
                    };
                }
                function evaluatePixel(sample) {
                    return [sample.B03 * 2.5, sample.B02 * 2.5, sample.B01 * 2.5];
                }
            """
        else:
            raise ValueError(f"Unsupported collection: {collection}")

    def request_data(self):
        """request true color image from Sentinel Hub, without clouds if time frame is at least a month"""
        # get selected file type, default is TIFF
        mime_type = self.mime_type_mapping.get(self.config.selected_file_type, MimeType.TIFF)

        # collection
        if self.config.additional_options:
            collection = self.collection_mapping.get(self.config.additional_options.collection)
        else:
            # default collection
            collection = DataCollection.SENTINEL2_L1C

        # evalscript
        evalscript_true_color = self.get_evalscript_for_collection(collection)

        # resolution
        resolution = self.resolution_mapping.get(collection)

        # coords
        coords = (
        self.config.coords[0].x(), self.config.coords[0].y(), self.config.coords[1].x(), self.config.coords[1].y())
        bbox = BBox(bbox=coords, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)

        request_true_color = SentinelHubRequest(
            data_folder=self.config.download_directory,
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=collection,
                    time_interval=(self.config.start_date, self.config.end_date),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", mime_type)],
            bbox=bbox,
            size=size,
            config=self.sh_config,
        )

        # check for download
        if self.config.download_checked:
            image_download = request_true_color.get_data(save_data=True)
            image = image_download[0]
            if not image_download:
                QgsMessageLog.logMessage("Image download failed!", level=Qgis.Critical)
                display_error_message("Image download failed!")

            # check for import
            if self.config.import_checked:
                # get path to file
                filename_list = request_true_color.get_filename_list()
                filename = filename_list[0]
                file_path = os.path.join(self.config.download_directory, filename)

                import_into_qgis(file_path, "Sentinel Hub Layer")

        else:
            image = request_true_color.get_data()[0]

            # check for import
            if self.config.import_checked:
                self.import_into_qgis_without_download(image, size, bbox)

        self.plot_image(image, 1, [0, 1])