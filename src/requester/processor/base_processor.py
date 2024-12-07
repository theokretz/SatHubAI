from abc import ABC, abstractmethod

import numpy as np
import rasterio.features
import matplotlib.pyplot as plt
import os
import uuid
from ...exceptions.ndvi_calculation_exception import NDVICalculationError

class Processor(ABC):
    def __init__(self, config, provider, collection):
        self._config = config
        self._provider = provider
        self._collection = collection

        self.cmap_mapping = {
            "Red": "Reds",
            "Green": "Greens",
            "Blue": "Blues",
            "Near Infrared": "plasma"
        }

    @abstractmethod
    def process(self, selected_item):
        """abstract method to process a collection"""
        pass

    def plot_image_single_band(self, asset_url, band):
        if self._config.plot_checked:
            with rasterio.open(asset_url) as src:
                image = src.read(1)

            cmap = self.cmap_mapping.get(band, "gray")

            plt.figure()
            plt.imshow(image, cmap=cmap)
            plt.title(self._provider.plot_title + " - " + band)
            plt.colorbar()
            plt.show()

    @staticmethod
    def get_unique_filename(base_directory, base_filename, file_type):
        unique_id = uuid.uuid4()
        unique_filename = f"{base_filename}_{unique_id}."+ file_type
        return os.path.join(base_directory, unique_filename)

    def save_image(self, asset_url, band=None):
        band = band.lower().replace(" ", "_")
        file_path = self.get_unique_filename(self._config.download_directory, self._provider.filename + "_" + band, self._config.selected_file_type)
        with rasterio.open(asset_url) as src:
            image = src.read()
            profile = src.profile
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(image)

    @staticmethod
    def normalize(array):
        return (array -  array.min()) / (array.max() - array.min())

    @staticmethod
    def create_valid_mask(red, green, blue, no_data_value=0):
        return (red != no_data_value) & (green != no_data_value) & (blue != no_data_value)

    @staticmethod
    def add_alpha_channel(true_color, valid_mask):
        """creates and adds alpha channel to the true color image, alpha channel is used to indicate transparency"""
        true_color_8bit = (true_color * 255).astype('uint8')

        alpha = np.where(valid_mask, 255, 0).astype('uint8')

        true_color_with_alpha = np.dstack((true_color_8bit, alpha))
        return true_color_with_alpha

    def true_false_color_calculation(self, red_url, green_url, blue_url, alpha_mask=False):
        with rasterio.open(red_url) as red_src, rasterio.open(green_url) as green_src, rasterio.open(blue_url) as blue_src:
            red = red_src.read(1).astype('float32')
            green = green_src.read(1).astype('float32')
            blue = blue_src.read(1).astype('float32')

            red_norm = self.normalize(red)
            green_norm = self.normalize(green)
            blue_norm = self.normalize(blue)

            if not alpha_mask:
                return np.dstack((red_norm, green_norm, blue_norm))
            else:
                true_color = np.dstack((red_norm, green_norm, blue_norm))
                valid_mask = self.create_valid_mask(red, green, blue)
                return self.add_alpha_channel(true_color, valid_mask)


    def ndvi_calculation(self, selected_item):
        if self._collection == "landsat-c2-l2" or self._collection == "landsat-c2-l1":
            red_url = selected_item.assets["red"].href
            nir_url = selected_item.assets["nir08"].href
        elif self._provider == self._provider.EARTH_SEARCH:
            red_url = selected_item.assets["red"].href
            nir_url = selected_item.assets["nir"].href
        elif self._provider == self._provider.PLANETARY_COMPUTER:
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

            if self._config.plot_checked:
                # plot NDVI
                plt.figure()
                plt.imshow(ndvi, cmap="RdYlGn")
                plt.colorbar(label="NDVI")
                plt.title("NDVI " + self._provider.plot_title + "-" + self._collection)
                plt.axis("off")
                plt.show()

            if self._config.download_checked:
                ndvi_path = self.get_unique_filename(self._config.download_directory, "ndvi_" + self._provider.filename + "_" + self._collection, self._config.selected_file_type)
                plt.savefig(ndvi_path, bbox_inches='tight', pad_inches=0, dpi=300)