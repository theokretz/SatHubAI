# stac_requester.py
# credits: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/

import rasterio.features
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import numpy as np
import os
import uuid
import rioxarray
import stackstac
from rasterio.plot import show
from osgeo import gdal

from ..exceptions.ndvi_calculation_exception import NDVICalculationError
from .requester import Requester
from .stac_service import StacService
from ..utils import import_into_qgis, display_warning_message


class StacRequester(Requester):
    def __init__(self, config, provider):
        super().__init__(config)
        self.config = config
        self.provider = provider

        self.collection_mapping = {
            "Sentinel-2 L1C": "sentinel-2-l1c" ,
            "Sentinel-2 L2A": "sentinel-2-l2a",
            "Landsat Collection 2 Level 1": "landsat-c2-l1",
            "Landsat Collection 2 Level 2": "landsat-c2-l2",
        }

    def plot_image(self, asset_url, scale_factor=0.1):
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
        plt.title(self.provider.plot_title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def get_unique_filename(base_directory, base_filename, file_type):
        unique_id = uuid.uuid4()
        unique_filename = f"{base_filename}_{unique_id}."+ file_type
        return os.path.join(base_directory, unique_filename)

    def save_image(self, asset_url):
        file_path = self.get_unique_filename(self.config.download_directory, self.provider.filename, self.config.selected_file_type)
        with rasterio.open(asset_url) as src:
            image = src.read()
            profile = src.profile
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(image)

    def ndvi_calculation(self, selected_item):
        if self.provider == self.provider.EARTH_SEARCH:
            red_url = selected_item.assets["red"].href
            nir_url = selected_item.assets["nir"].href
        elif self.provider == self.provider.PLANETARY_COMPUTER:
            red_url = selected_item.assets["B04"].href
            nir_url = selected_item.assets["B08"].href
        else:
            raise NDVICalculationError("NDVI can not be calculated for this asset.")

        if not red_url or not nir_url:
            raise NDVICalculationError("NDVI can not be calculated for this asset.")

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
            plt.title("NDVI " + self.provider.plot_title)
            plt.axis("off")
            plt.show()

            if self.config.download_checked:
                ndvi_path = self.get_unique_filename(self.config.download_directory, "ndvi_" + self.provider.filename, self.config.selected_file_type)
                plt.savefig(ndvi_path, bbox_inches='tight', pad_inches=0, dpi=300)


    def process_sentinel_2_l1c(self, selected_item):
        os.environ["AWS_REQUEST_PAYER"] = "requester"
        os.environ["AWS_PROFILE"] = "default"
        data_array = stackstac.stack(items=selected_item, dtype="float64", resolution=10, rescale=False)

        da_l1c = data_array.sel(band=["red", "green", "blue"]).squeeze()

        da_l1c.astype("int").plot.imshow(rgb="band", robust=True)
        plt.title("Sentinel-2 L1C Earth Search")
        plt.show()

        if self.config.download_checked:
            file_path = self.get_unique_filename(self.config.download_directory, self.provider.filename, self.config.selected_file_type)
            da_l1c.rio.to_raster(file_path)

        if self.config.import_checked:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                temp_path = temp_file.name
                da_l1c.rio.to_raster(temp_path, driver="GTiff", crs="EPSG:32633")
                import_into_qgis(temp_path, "Sentinel-2 L1C Earth Search")

        if self.config.additional_options and self.config.additional_options.ndvi_checked.isChecked():
               self.ndvi_calculation(selected_item)

    @staticmethod
    def calculate_bbox(coords):
        # get min and max coordinates
        min_lon = min(coords[0].x(), coords[1].x())
        max_lon = max(coords[0].x(), coords[1].x())
        min_lat = min(coords[0].y(), coords[1].y())
        max_lat = max(coords[0].y(), coords[1].y())
        return min_lon, min_lat, max_lon, max_lat

    def request_data(self):
        catalog = StacService.get_client(self.provider)

        collections = catalog.get_collections()
        for collection in collections:
            print(collection.id)

        bbox = self.calculate_bbox(self.config.coords)

        if self.config.additional_options:
            collection =  self.collection_mapping.get(self.config.additional_options.collection)
        else:
            collection = "sentinel-2-l1c"

        # Prepare search parameters
        search_params = {
            "collections": collection,
            "bbox": bbox,
            "datetime": f"{self.config.start_date}/{self.config.end_date}",
        }

        # add query parameter only if supported - only supported in sentinel-2-l2a collection
        if collection == "sentinel-2-l2a":
            search_params["query"] = {
                "eo:cloud_cover": {"lt": 10},
                "s2:nodata_pixel_percentage": {"lt": 1},
            }


        search = catalog.search(**search_params)
        items = search.item_collection()
        print("Collection size: ", len(items))

        if not items:
            display_warning_message("Change your options.", "No satellite data found!")
            return

        # select item with the lowest cloudiness -> problem: often selects image with no data areas
        selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
        print(selected_item.assets)
        asset_url = selected_item.assets["visual"].href
        print(asset_url)

        if collection == "sentinel-2-l1c":
            self.process_sentinel_2_l1c(selected_item)
        else:
            self.plot_image(asset_url)

            if self.config.import_checked:
                    import_into_qgis(asset_url, self.provider.qgis_layer_name)

            if self.config.download_checked:
                self.save_image(asset_url)

            if self.config.additional_options and self.config.additional_options.ndvi_checked.isChecked():
               self.ndvi_calculation(selected_item)