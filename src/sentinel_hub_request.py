# sentinel_hub_request.py
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

from .utils import import_into_qgis, display_error_message

sh_config = SHConfig()

if not sh_config.sh_client_id or not sh_config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

resolution = 60

mime_type_mapping = {
    "TIFF": MimeType.TIFF,
    "PNG": MimeType.PNG,
    "JPEG": MimeType.JPG
}

def plot_image(image, factor=1.0, clip_range=None, normalize=True):
    if normalize:
        image= (image - image.min()) / (image.max() - image.min())
    image *= factor
    if clip_range:
        image = np.clip(image, clip_range[0], clip_range[1])
    plt.figure()
    plt.imshow(image)
    plt.title("Sentinel Hub")
    plt.axis('off')
    plt.show()

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
                transform=from_bounds(lower_left[0], lower_left[1], upper_right[0], upper_right[1], width=size[0], height=size[1]),
                crs="EPSG:4326"
        ) as tmp_dataset:
            tmp_dataset.write(image.transpose(2, 0, 1))

        import_into_qgis(tmp_file.name, "Sentinel Hub Layer")

def true_color_sentinel_hub(config):
    """request true color image from Sentinel Hub, without clouds if time frame is at least a month"""
    # get selected file type, default is TIFF
    mime_type = mime_type_mapping.get(config.selected_file_type, MimeType.TIFF)

    # coords
    coords = (config.coords[0].x(), config.coords[0].y(), config.coords[1].x(), config.coords[1].y())
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)


    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        data_folder=config.download_directory,
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(config.start_date, config.end_date),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mime_type)],
        bbox=bbox,
        size=size,
        config=sh_config,
    )

    # check for download
    if config.download_checked:
        image_download = request_true_color.get_data(save_data=True)
        image = image_download[0]
        if not image_download:
            QgsMessageLog.logMessage("Image download failed!", level=Qgis.Critical)
            display_error_message("Image download failed!")

        # check for import
        if config.import_checked:
            # get path to file
            filename_list = request_true_color.get_filename_list()
            filename = filename_list[0]
            file_path = os.path.join(config.download_directory, filename)

            import_into_qgis(file_path, "Sentinel Hub Layer")

    else:
        image = request_true_color.get_data()[0]

        # check for import
        if config.import_checked:
            import_into_qgis_without_download(image, size, bbox)


    plot_image(image, 1, [0,1])




