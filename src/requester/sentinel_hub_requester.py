# sentinel_hub_requester.py
# credits: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html
import os
import tarfile

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
from ..utils import import_into_qgis, display_error_message, display_warning_message


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

        self.band_mapping = {
            "Sentinel-2 L1C": {
                "True Color": ["B04", "B03", "B02"],  # RGB
                "False Color": ["B08", "B04", "B03"],  # NIR, Red, Green
                "Red": ["B04"],
                "Green": ["B03"],
                "Blue": ["B02"],
                "Near Infrared": ["B08"]
            },
            "Sentinel-2 L2A": {
                "True Color": ["B04", "B03", "B02"],  # RGB
                "False Color": ["B08", "B04", "B03"],  # NIR, Red, Green
                "Red": ["B04"],
                "Green": ["B03"],
                "Blue": ["B02"],
                "Near Infrared": ["B08"]
            },
            "Landsat 1-5 MSS L1": {
                "False Color": ["B04", "B02", "B01"],  # NIR, Red, Green
                "Red": ["B02"],
                "Green": ["B01"],
                "Near Infrared": ["B04"]
            },
            "Landsat 4-5 TM L1": {
                "True Color": ["B03", "B02", "B01"],  # RGB
                "False Color": ["B04", "B03", "B02"],  # NIR, Red, Green
                "Red": ["B03"],
                "Green": ["B02"],
                "Blue": ["B01"],
                "Near Infrared": ["B04"]
            },
            "Landsat 4-5 TM L2": {
                "True Color": ["B03", "B02", "B01"],  # RGB
                "False Color": ["B04", "B03", "B02"],  # NIR, Red, Green
                "Red": ["B03"],
                "Green": ["B02"],
                "Blue": ["B01"],
                "Near Infrared": ["B04"]
            },
            "Landsat 7 ETM+ L1": {
                "True Color": ["B03", "B02", "B01"],  # RGB
                "False Color": ["B04", "B03", "B02"],  # NIR, Red, Green
                "Red": ["B03"],
                "Green": ["B02"],
                "Blue": ["B01"],
                "Near Infrared": ["B04"]
            },
            "Landsat 7 ETM+ L2": {
                "True Color": ["B03", "B02", "B01"],  # RGB
                "False Color": ["B04", "B03", "B02"],  # NIR, Red, Green
                "Red": ["B03"],
                "Green": ["B02"],
                "Blue": ["B01"],
                "Near Infrared": ["B04"]
            },
            "Landsat 8-9 OLI/TIRS L1": {
                "True Color": ["B04", "B03", "B02"],  # RGB
                "False Color": ["B05", "B04", "B03"],  # NIR, Red, Green
                "Red": ["B04"],
                "Green": ["B03"],
                "Blue": ["B02"],
                "Near Infrared": ["B05"]
            },
            "Landsat 8-9 OLI/TIRS L2": {
                "True Color": ["B04", "B03", "B02"],  # RGB
                "False Color": ["B05", "B04", "B03"],  # NIR, Red, Green
                "Red": ["B04"],
                "Green": ["B03"],
                "Blue": ["B02"],
                "Near Infrared": ["B05"]
            }
        }

    @staticmethod
    def plot_image(image, title, band_name):
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def import_into_qgis_without_download(image, size, bbox, title):
        """load a satellite image into QGIS without downloading it first via temporary file."""

        # determine the number of bands
        if image.ndim == 2:  # single band image
            count = 1
            data = image[np.newaxis, ...] # add a new axis for the band dimension
        elif image.ndim == 3:  # Multi-band image
            count = image.shape[2]
            data = image.transpose(2, 0, 1)  # transpose for rasterio (bands, rows, cols)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            with rasterio.open(
                    tmp_file.name, 'w',
                    driver='GTiff',
                    width=size[0],
                    height=size[1],
                    count=count,
                    dtype=image.dtype,
                    transform=from_bounds(bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1],
                    width=size[0],
                    height=size[1]),
            ) as tmp_dataset:
                tmp_dataset.write(data)

            import_into_qgis(tmp_file.name, title)


    def generate_evalscript(self, bands, collection_name):
        """generates an evalscript"""
        input_bands = set()
        outputs = []
        evaluations = []
        count = 0

        # first band needs the default identifier
        for band_name in bands:
            sentinel_bands = self.band_mapping[collection_name][band_name]
            if sentinel_bands is None:
                raise ValueError(f"Unknown band name: {band_name}")

            input_bands.update(sentinel_bands)

            if len(sentinel_bands) == 3:  # True Color or False Color
                if count == 0:
                    outputs.append(f"""
                       {{
                           id: "default",
                           bands: 3,
                           sampleType: "AUTO"
                       }}
                       """)
                    evaluations.append(
                        f'"default": [samples.{sentinel_bands[0]} * 2.5, samples.{sentinel_bands[1]} * 2.5, samples.{sentinel_bands[2]} * 2.5]')
                else:
                    outputs.append(f"""
                       {{
                           id: "{band_name.lower().replace(' ', '_')}",
                           bands: 3,
                           sampleType: "AUTO"
                       }}
                       """)
                    evaluations.append(
                        f'"{band_name.lower().replace(" ", "_")}": [samples.{sentinel_bands[0]} * 2.5, samples.{sentinel_bands[1]} * 2.5, samples.{sentinel_bands[2]} * 2.5]')
            elif len(sentinel_bands) == 1:  # Single-band outputs
                if count == 0:
                    outputs.append(f"""
                       {{
                           id: "default",
                           bands: 1,
                           sampleType: "AUTO"
                       }}
                       """)
                    evaluations.append(f'"default": [samples.{sentinel_bands[0]}]')
                else:
                    outputs.append(f"""
                       {{
                           id: "{band_name.lower().replace(' ', '_')}",
                           bands: 1,
                           sampleType: "AUTO"
                       }}
                       """)
                    evaluations.append(f'"{band_name.lower().replace(" ", "_")}": [samples.{sentinel_bands[0]}]')
            count += 1

        # generate evalscript
        evalscript = f"""
           //VERSION=3
           function setup() {{
               return {{
                   input: [{{
                       bands: [{", ".join(f'"{band}"' for band in input_bands)}]
                   }}],
                   output: [
                       {", ".join(outputs)}
                   ]
               }};
           }}

           function evaluatePixel(samples) {{
               return {{
                   {", ".join(evaluations)}
               }};
           }}
           """
        return evalscript

    def generate_responses(self, bands, mime_type, collection_name):
        """
        Generates a list of SentinelHubRequest.output_response objects based on user-selected bands.

        Parameters:
            bands (list): List of user-friendly band names.
            mime_type(MimeType): Requested file format.
            collection_name: User-friendly collection name.

        Returns:
            list: A list of SentinelHubRequest.output_response objects.
        """
        # first one is always default
        responses = [SentinelHubRequest.output_response("default", mime_type)]

        # skip the first one - it's already added as default
        for band_name in bands[1:]:
            sentinel_bands = self.band_mapping[collection_name][band_name]
            if sentinel_bands is None:
                raise ValueError(f"Unknown band name: {band_name}")

            responses.append(SentinelHubRequest.output_response(f"{band_name.lower().replace(' ', '_')}", mime_type))

        return responses

    def request_data(self):
        """request true color image from Sentinel Hub, without clouds if time frame is at least a month"""

        # get selected file type, default is TIFF
        mime_type = self.mime_type_mapping.get(self.config.selected_file_type, "TIFF")

        # collection
        if self.config.additional_options:
            collection = self.collection_mapping.get(self.config.additional_options.collection)
            collection_name = self.config.additional_options.collection
        else:
            # default collection
            collection = DataCollection.SENTINEL2_L1C
            collection_name = "Sentinel-2 L1C"

        # bands - default is True Color
        if self.config.additional_options:
            bands = self.config.additional_options.bands
        else:
            bands = ["True Color"]

        # evalscript
        evalscript = self.generate_evalscript(bands, collection_name)

        responses = self.generate_responses(bands, mime_type, collection_name)

        # resolution
        resolution = self.resolution_mapping.get(collection)

        # coords
        coords = (self.config.coords[0].x(), self.config.coords[0].y(), self.config.coords[1].x(), self.config.coords[1].y())
        bbox = BBox(bbox=coords, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)


        request_true_color = SentinelHubRequest(
            data_folder=self.config.download_directory,
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=collection,
                    time_interval=(self.config.start_date, self.config.end_date),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                )
            ],
            responses=responses,
            bbox=bbox,
            size=size,
            config=self.sh_config,
        )

        images = request_true_color.get_data()[0]

        # if it is just a single image - transform it to dict
        if isinstance(images, np.ndarray):
            images = {"default": images}

        # help counter to access bands array
        count = 0
        for filename, image in images.items():
            band_name = bands[count]

            # check if image is black/blank
            if np.all(image == 0):
                display_warning_message("The Satellite Image is blank." , "No Satellite Image!")
                return

            # check for import and no download
            if self.config.import_checked and not self.config.download_checked:
                self.import_into_qgis_without_download(image, size, bbox, f"Sentinel Hub {collection_name} - {band_name}")

            self.plot_image(image, f"Sentinel Hub {collection_name} - {band_name}", band_name)
            count += 1


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

                # .tar directory gets created for multiple images
                if len(bands) > 1:
                    extract_directory = os.path.dirname(file_path)

                    # extract images
                    with tarfile.open(file_path, "r") as tar:
                        tar.extractall(path=extract_directory)

                        for file in os.listdir(extract_directory):
                            if self.config.selected_file_type == "TIFF":
                                file_ending = ".tif"
                            elif self.config.selected_file_type == "JPEG":
                                file_ending = ".jpg"
                            elif self.config.selected_file_type == "PNG":
                                file_ending = ".png"
                            else:
                                raise ValueError(f"Unsupported file type: {self.config.selected_file_type}")

                            # go through all files and import into qgis
                            if file.endswith(file_ending):
                                full_path = os.path.join(extract_directory, file)
                                import_into_qgis(full_path, f"Sentinel Hub {collection_name} - {file}")




