# stac_request.py
# credits: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/

import rasterio.features
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import numpy as np
import os
import uuid

from .stac_service import StacService
from .utils import import_into_qgis, display_error_message


def plot_image(asset_url, title, scale_factor=0.1, ):
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


def get_unique_filename(base_directory, base_filename, file_type):
    unique_id = uuid.uuid4()
    unique_filename = f"{base_filename}_{unique_id}."+ file_type
    return os.path.join(base_directory, unique_filename)


def save_image(asset_url, directory, file_type, filename):
    file_path = get_unique_filename(directory, filename,file_type)
    with rasterio.open(asset_url) as src:
        image = src.read()
        profile = src.profile
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(image)


def ndvi_calculation(selected_item, download_checked, directory, file_type, stac_provider):
    if stac_provider == stac_provider.AWS:
        red_url = selected_item.assets["red"].href
        nir_url = selected_item.assets["nir"].href
    elif stac_provider == stac_provider.PLANETARY_COMPUTER:
        red_url = selected_item.assets["B04"].href
        nir_url = selected_item.assets["B08"].href
    else:
        display_error_message("NDVI can not be calculated for this asset.")
        return


    if not red_url or not nir_url:
        display_error_message("NDVI can not be calculated for this asset.")


    with (rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src):
        red = red_src.read(1).astype(float)
        nir = nir_src.read(1).astype(float)

        # calculate NDVI
        ndvi = (nir - red + 1e-10) / (nir + red + 1e-10)    # prevent division by zero
        ndvi = np.clip(ndvi, -1, 1)


        # plot NDVI
        plt.figure()
        plt.imshow(ndvi, cmap="RdYlGn")
        plt.colorbar(label="NDVI")
        plt.title("NDVI " + stac_provider.plot_title)
        plt.axis("off")
        plt.show()

        if download_checked:
            ndvi_path = get_unique_filename(directory, "ndvi_" + stac_provider.filename,file_type)
            plt.savefig(ndvi_path, bbox_inches='tight', pad_inches=0, dpi=300)




def true_color_stac(coords_wgs84, start_date, end_date, download_checked, file_type, directory, import_checked, ndvi_checked, stac_provider):

    catalog = StacService.get_client(stac_provider)

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

    plot_image(asset_url, stac_provider.plot_title)

    if import_checked:
        import_into_qgis(asset_url, stac_provider.qgis_layer_name)

    if download_checked:
        save_image(asset_url, directory, file_type, stac_provider.filename)

    if ndvi_checked:
       ndvi_calculation(selected_item, download_checked, directory, file_type, stac_provider)


