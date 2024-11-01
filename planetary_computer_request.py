# planetary_computer_request.py
# credits: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/

import rasterio.features
import matplotlib.pyplot as plt
import planetary_computer
from pystac_client import Client
from rasterio.enums import Resampling
from qgis._core import QgsProject, QgsMessageLog, Qgis
from qgis.core import QgsRasterLayer
import os
import uuid

from .utils import display_error_message


def plot_image(asset_url, scale_factor=0.1, title="Planetary Computer"):
    with rasterio.open(asset_url) as src:
        image = src.read(
            [1, 2, 3],  # RGB bands
            out_shape=(
                src.count,
                # scale it down so it loads faster
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )

    # rearranges dimensions
    image = image.transpose((1, 2, 0))

    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def get_unique_filename(base_directory, base_filename="planetary_computer"):
    unique_id = uuid.uuid4()
    unique_filename = f"{base_filename}_{unique_id}."
    return os.path.join(base_directory, unique_filename)


def save_image(asset_url, directory, file_type):
    file_path = get_unique_filename(directory) + file_type
    with rasterio.open(asset_url) as src:
        image = src.read()
        profile = src.profile
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(image)

    # TODO: add checkbox for this
    import_into_qgis(file_path)


# TODO: put into utils
def import_into_qgis(file_path):
    # load the raster layer into QGIS
    raster_layer = QgsRasterLayer(file_path, "Planetary Computer Image")
    if not raster_layer.isValid():
        QgsMessageLog.logMessage("Layer failed to load!", level=Qgis.Critical)
        display_error_message("Image Layer failed to load!")
    else:
        QgsProject.instance().addMapLayer(raster_layer)


def true_color_planetary_computer(coords_wgs84, start_date, end_date, download_checked, file_type, directory):
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        # automatically sign the STAC metadata, so that assets can be accessed
        modifier=planetary_computer.sign_inplace,
    )

    # get min and max coordinates
    min_lon = min(coords_wgs84[0].x(), coords_wgs84[1].x())
    max_lon = max(coords_wgs84[0].x(), coords_wgs84[1].x())
    min_lat = min(coords_wgs84[0].y(), coords_wgs84[1].y())
    max_lat = max(coords_wgs84[0].y(), coords_wgs84[1].y())
    bbox = (min_lon, min_lat, max_lon, max_lat)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            # filter for minimal cloudiness
            "eo:cloud_cover": {"lt": 10},
            # fix -> filter for minimal no data areas
            "s2:nodata_pixel_percentage": {"lt": 1},
        },
    )

    items = search.item_collection()

    # select item with the lowest cloudiness -> problem: often selects image with no data areas
    selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
    asset_url = selected_item.assets["visual"].href

    plot_image(asset_url)

    if download_checked:
        save_image(asset_url, directory, file_type)