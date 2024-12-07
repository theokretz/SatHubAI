import numpy as np

from .base_processor import Processor
import os
import rasterio.features
import matplotlib.pyplot as plt
import tempfile
from ...utils import import_into_qgis


class LandsatProcessor(Processor):
    def __init__(self, config, provider, collection):
        super().__init__(config, provider, collection)
        self._config = config
        self._provider = provider
        self._collection = collection

    def process(self, selected_item):
        """Implements processing for Landsat collections."""
        if self._collection == "landsat-c2-l1" or self._collection == "landsat-c2-l2":
            self.process_landsat_c2(selected_item)
        else:
            raise ValueError(f"Unsupported Landsat collection: {self._collection}")

    @staticmethod
    def write_temporary_file_single_band(asset_url):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            with rasterio.open(asset_url) as src:
                with rasterio.open(
                        temp_file.name,
                        'w',
                        driver='GTiff',
                        height=src.height,
                        width=src.width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=src.transform,
                ) as dst:
                    dst.write(src.read())
            return temp_file.name

    @staticmethod
    def write_temporary_file_multiple_bands(src, data):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            with rasterio.open(
                    temp_file.name,
                    'w',
                    driver='GTiff',
                    height=src.height,
                    width=src.width,
                    count=4,
                    dtype='uint8',
                    crs=src.crs,
                    transform=src.transform,
            ) as dst:
                dst.write(data[:, :, 0], 1)
                dst.write(data[:, :, 1], 2)
                dst.write(data[:, :, 2], 3)
                dst.write(data[:, :, 3], 4)
        return temp_file.name

    def save_image_multiple_bands(self, src, data, filename):
        file_path = self.get_unique_filename(self._config.download_directory, filename, self._config.selected_file_type)
        with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=4,
                dtype='uint8',
                crs=src.crs,
                transform=src.transform,
        ) as dst:
            dst.write(data[:, :, 0], 1)
            dst.write(data[:, :, 1], 2)
            dst.write(data[:, :, 2], 3)
            dst.write(data[:, :, 3], 4)
        return file_path


    def process_landsat_c2(self, selected_item):
        os.environ["AWS_REQUEST_PAYER"] = "requester"
        os.environ["AWS_PROFILE"] = "default"

        if self._config.additional_options:
            bands = self._config.additional_options.bands
        else:
            if self._collection == "landsat-c2-l1":
                bands = ["False Color"]
            elif self._collection == "landsat-c2-l2":
                bands = ["True Color"]
            else:
                raise ValueError(f"Unsupported Landsat collection: {self._collection}")

        for band in bands:
            if band == "True Color" or band == "False Color":
                if band == "True Color":
                    red_name = self._provider.collection_mapping[self._collection]["Red"]
                    green_name = self._provider.collection_mapping[self._collection]["Green"]
                    blue_name = self._provider.collection_mapping[self._collection]["Blue"]
                else:
                    red_name = self._provider.collection_mapping[self._collection]["Near Infrared"]
                    green_name = self._provider.collection_mapping[self._collection]["Red"]
                    blue_name = self._provider.collection_mapping[self._collection]["Green"]

                red_url = selected_item.assets[red_name].href
                green_url = selected_item.assets[green_name].href
                blue_url = selected_item.assets[blue_name].href

                true_color_with_alpha = self.true_false_color_calculation(red_url, green_url, blue_url, True)

                # import
                if self._config.plot_checked:
                    plt.figure()
                    plt.imshow(true_color_with_alpha.astype('uint8'))
                    plt.title(self._provider.plot_title + " " + self._collection + "-" + band)
                    plt.axis("off")
                    plt.show()

                # download
                if self._config.download_checked:
                    with rasterio.open(red_url) as red_src:
                        file_path = self.save_image_multiple_bands(red_src, true_color_with_alpha, f"{self._provider.filename}_{self._collection}_{band.lower().replace(' ', '_')}")
                        # import
                        if self._config.import_checked:
                            import_into_qgis(file_path, self._provider.qgis_layer_name + " " + self._collection + "-" + band)

                # just import no download
                if self._config.import_checked and not self._config.download_checked:
                    with rasterio.open(red_url) as red_src:
                        filepath = self.write_temporary_file_multiple_bands(red_src, true_color_with_alpha)
                        import_into_qgis(filepath, self._provider.qgis_layer_name + " " + self._collection + "-" + band)
            else:
                # handle single bands
                asset_name = self._provider.collection_mapping[self._collection][band]
                asset_url = selected_item.assets[asset_name].href
                if self._config.plot_checked:
                    self.plot_image_single_band(asset_url, band)
                if self._config.download_checked:
                    self.save_image(asset_url, band)
                if self._config.import_checked:
                    if self._provider == self._provider.PLANETARY_COMPUTER:
                        import_into_qgis(asset_url, band)
                    else:
                        filepath = self.write_temporary_file_single_band(asset_url)
                        import_into_qgis(filepath, self._provider.qgis_layer_name + " " + self._collection + "-" + band)

        # ndvi
        if self._config.additional_options and self._config.additional_options.ndvi_checked:
            self.ndvi_calculation(selected_item)