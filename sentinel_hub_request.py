# sentinel_hub_request.py
# Most of this code is from: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html
import matplotlib.pyplot as plt
import numpy as np
from qgis._core import QgsProject, QgsMessageLog, Qgis
from qgis.core import QgsRasterLayer
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

config = SHConfig()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

vienna_coords_wgs84 = (16.280155, 48.151886, 16.466296, 48.260341)
resolution = 60
vienna_bbox = BBox(bbox=vienna_coords_wgs84, crs=CRS.WGS84)
vienna_size = bbox_to_dimensions(vienna_bbox, resolution=resolution)

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
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def true_color_with_cloud_mask(start_date, end_date):
    evalscript_clm = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02", "B03", "B04", "CLM"],
        output: { bands: 3 }
      }
    }

    function evaluatePixel(sample) {
      if (sample.CLM == 1) {
        return [0.75 + sample.B04, sample.B03, sample.B02]
      }
      return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];
    }
    """

    request_true_color = SentinelHubRequest(
        data_folder="test_dir",
        evalscript=evalscript_clm,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=vienna_bbox,
        size=vienna_size,
        config=config,
    )
    data_with_cloud_mask_request = request_true_color.get_data(save_data=True)
    data_with_cloud_mask = data_with_cloud_mask_request[0]
    plot_image(data_with_cloud_mask, 1, [0,1])


def true_color_without_clouds(start_date, end_date, download_checked, selected_file_type, coords_wgs84):
    """request true color image from Sentinel Hub, without clouds if time frame is at least a month"""
    # get selected file type, default is TIFF
    mime_type = mime_type_mapping.get(selected_file_type, MimeType.TIFF)

    # coords
    coords = (coords_wgs84[0].x(), coords_wgs84[0].y(), coords_wgs84[1].x(), coords_wgs84[1].y())
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
        data_folder="test_dir",
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mime_type)],
        bbox=bbox,
        size=size,
        config=config,
    )

    # check for download
    if download_checked:
        image_download = request_true_color.get_data(save_data=True)
        image = image_download[0]
        if not image_download:
            print("Image download failed!")
            QgsMessageLog.logMessage("Image download failed!", level=Qgis.Critical)
    else:
        image = request_true_color.get_data()[0]

    plot_image(image, 1, [0,1])



def import_into_qgis():
    # path to image, right now just a test path
    image_path = 'test_dir/c8088537648609eb3c59904559fcf26c/response.tiff'

    # load the raster layer into QGIS
    raster_layer = QgsRasterLayer(image_path, "Vienna Sentinel Image")
    if not raster_layer.isValid():
        QgsMessageLog.logMessage("Layer failed to load!", level=Qgis.Critical)
        print("File path:", image_path)
    else:
        QgsProject.instance().addMapLayer(raster_layer)

