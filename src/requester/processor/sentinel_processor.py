import numpy as np
import stackstac

from .base_processor import Processor
from ...utils import import_into_qgis
import rasterio.features
import matplotlib.pyplot as plt
import os
import rioxarray


class SentinelProcessor(Processor):
    def __init__(self, config, provider, collection):
        super().__init__(config, provider, collection)
        self._collection = collection
        self._provider = provider

    def process(self, selected_item):
        if self._collection == "sentinel-2-l2a":
            self.process_sentinel_2_l2a(selected_item)
        elif self._collection == "sentinel-2-l1c":
            self.process_sentinel_2_l1c(selected_item)
        else:
            raise ValueError(f"Unsupported collection: {self._collection}")

    def plot_image(self, asset_url, band):
        if self._config.plot_checked:
            with rasterio.open(asset_url) as src:
                image = src.read([1, 2, 3])

            # rearranges dimensions
            image = image.transpose((1, 2, 0))

            # plot the image
            plt.figure()
            plt.imshow(image)
            plt.title(self._provider.plot_title + " " + self._collection + " " + band)
            plt.show()

    def process_sentinel_2_l2a(self, selected_item):
        # default is True Color
        if self._config.additional_options:
            bands = self._config.additional_options.bands
        else:
            bands = ["True Color"]

        for band in bands:
            asset_name = self._provider.collection_mapping[self._collection][band]
            asset_url = selected_item.assets[asset_name].href

            if band == "True Color":
                self.plot_image(asset_url, band)
            elif band == "False Color":
                #TODO: implement false color
                pass
            else:
                self.plot_image_single_band(asset_url, band)

            if self._config.import_checked:
                import_into_qgis(asset_url, self._provider.qgis_layer_name)

            if self._config.download_checked:
                self.save_image(asset_url, band)

        if self._config.additional_options and self._config.additional_options.ndvi_checked.isChecked():
            self.ndvi_calculation(selected_item)

    def save_and_import_sentinel_2_l1c(self, data, band):
        """Saves and imports sentinel-2-l1c data into QGIS."""
        if self._config.download_checked:
            file_path = self.get_unique_filename(self._config.download_directory,
                                                 f"{self._provider.filename}_{self._collection}_{band.lower().replace(' ', '_')}",
                                                 self._config.selected_file_type)
            data.rio.to_raster(file_path)

        if self._config.import_checked:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                temp_path = temp_file.name
                data.rio.to_raster(temp_path, driver="GTiff", crs="EPSG:32633")
                import_into_qgis(temp_path, f"Earth Search Sentinel-2 L1C - {band}")

    def process_sentinel_2_l1c(self, selected_item):
        """processes the sentinel-2 l1c collection only available on Earth Search"""
        os.environ["AWS_REQUEST_PAYER"] = "requester"
        os.environ["AWS_PROFILE"] = "default"

        # default is true color
        if self._config.additional_options:
            bands = self._config.additional_options.bands
        else:
            bands = ["True Color"]

        data_array = stackstac.stack(items=selected_item, dtype=np.float64, resolution=10, rescale=False)

        for band in bands:
            if band == "True Color":
                print(band)
                selected_bands = ["red", "green", "blue"]
                da_l1c = data_array.sel(band=selected_bands).squeeze()

                if self._config.plot_checked:
                    da_l1c.astype("int").plot.imshow(rgb="band", robust=True)
                    plt.title("Earth Search Sentinel-2 L1C - True Color")
                    plt.show()

                self.save_and_import_sentinel_2_l1c(da_l1c, band)
            elif band == "False Color":
                selected_bands = ["nir", "red", "green"]
                da_l1c = data_array.sel(band=selected_bands).squeeze()

                if self._config.plot_checked:
                    plt.figure()
                    da_l1c.astype("int").plot.imshow(rgb="band", robust=True)
                    plt.title("Earth Search Sentinel-2 L1C - False Color")
                    plt.show()

                self.save_and_import_sentinel_2_l1c(da_l1c, band)

            else:
                band_da = data_array.sel(band=self._provider.collection_mapping[self._collection][band]).squeeze()

                if self._config.plot_checked:
                    plt.figure()
                    band_da.astype("int").plot.imshow(cmap=self.cmap_mapping.get(band), robust=True)
                    plt.title(f"Earth Search Sentinel-2 L1C - {band}")
                    plt.show()

                self.save_and_import_sentinel_2_l1c(band_da, band)

        if self._config.additional_options and self._config.additional_options.ndvi_checked.isChecked():
            self.ndvi_calculation(selected_item)