"""
base_processor
=================
Defines the abstract Processor class. The superclass of LandsatProcessor and SentinelProcessor.
"""
from abc import ABC, abstractmethod

import numpy as np
import rasterio.features
import matplotlib.pyplot as plt
import os
import uuid
import tempfile
from ...exceptions.ndvi_calculation_exception import NDVICalculationError
from ...utils import import_into_qgis


class Processor(ABC):
    """
    Abstract base class for processing satellite imagery data.
    It includes common methods for calculating NDVI, plotting images,
    generating true/False compositions and saving data to files.

    Parameters
    ----------
    config : RequestConfig
        Configuration containing user options.
    provider : Provider
        Satellite Data Provider, either PLANETARY_COMPUTER or EARTH_SEARCH.
    collection : str
        Satellite data collection.
    """
    def __init__(self, config, provider, collection):
        self._config = config
        self._provider = provider
        self._collection = collection

    @abstractmethod
    def process(self, selected_item):
        """
        Abstract method to process a specific item in the collection.

        Parameters
        ----------
        selected_item : pystac.item.Item
            STAC item to be processed.
        """
        pass

    def plot_image_single_band(self, asset_url, band):
        """
        Plots a single-band image.

        Parameters
        ----------
        asset_url : str
            URL to the image asset.
        band : str
            Band name to be visualized.
        """
        if self._config.plot_checked:
            with rasterio.open(asset_url) as src:
                image = src.read(1)

            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title(self._provider.plot_title + " - " + band)
            plt.colorbar()
            plt.show()

    @staticmethod
    def get_unique_filename(base_directory, base_filename, file_type):
        """
        Generates a unique filename.

        Parameters
        ----------
        base_directory : str
            Directory where the file will be saved.
        base_filename : str
            Base name for the file.
        file_type : str
            File extension/type.

        Returns
        -------
        str
            Full path to the unique file.
        """
        unique_id = uuid.uuid4()
        unique_filename = f"{base_filename}_{unique_id}."+ file_type
        return os.path.join(base_directory, unique_filename)

    def save_image_single_band(self, asset_url, band):
        """
        Saves a single-band image.

        Parameters
        ----------
        asset_url : str
            URL to the image asset.
        band : str
            Band name to be included in the filename.
        """
        band = band.lower().replace(" ", "_")
        file_path = self.get_unique_filename(self._config.download_directory, self._provider.filename + "_" + band, self._config.selected_file_type)
        with rasterio.open(asset_url) as src:
            image = src.read()
            profile = src.profile
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(image)

    @staticmethod
    def normalize(image):
        """
        Normalizes an image to the range [0, 1].

        Parameters
        ----------
        image : numpy.ndarray
            Input image to be normalized.

        Returns
        -------
        numpy.ndarray
            Normalized image.
        """
        return (image - image.min()) / (image.max() - image.min())

    @staticmethod
    def create_valid_mask(red, green, blue, no_data_value=0):
        """
        Creates a valid pixel mask based on RGB values.

        Parameters
        ----------
        red : numpy.ndarray
            Red band data.
        green : numpy.ndarray
            Green band data.
        blue : numpy.ndarray
            Blue band data.
        no_data_value : int, optional
            Value indicating invalid pixels (default is 0).

        Returns
        -------
        numpy.ndarray
            Boolean mask of valid pixels.
        """
        return (red != no_data_value) & (green != no_data_value) & (blue != no_data_value)

    @staticmethod
    def add_alpha_channel(true_color, valid_mask):
        """
        Creates and adds an alpha channel to the true color image, an alpha channel is used to indicate transparency

        Parameters
        ----------
        true_color : numpy.ndarray
            True-color image (3D array).
        valid_mask : numpy.ndarray
            Boolean mask for valid pixels.
        """
        true_color_8bit = (true_color * 255).astype('uint8')
        alpha = np.where(valid_mask, 255, 0).astype('uint8')
        return np.dstack((true_color_8bit, alpha))


    def true_false_color_calculation(self, red_url, green_url, blue_url, alpha_mask=False):
        """
        Calculates true/false color composites.

        Parameters
        ----------
        red_url : str
            URL to the red band.
        green_url : str
            URL to the green band.
        blue_url : str
            URL to the blue band.
        alpha_mask : bool, optional
            Whether to add an alpha mask (default is False).

        Returns
        -------
        numpy.ndarray
            RGB or RGBA image.
        """
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

    @staticmethod
    def write_bands_to_file(output_path, src, data, count, dtype):
        """
        Writes multi-band data to a file.

        Parameters
        ----------
        output_path : str
            Path to the output file.
        src : rasterio.io.DatasetReader
            Source dataset for metadata.
        data : numpy.ndarray
            Multi-band data to be written.
        count : int
            Number of bands.
        dtype : str
            Data type of the bands.
        """
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=count,
            dtype=dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            if count > 1:
                for i in range(count):
                    dst.write(data[:, :, i], i + 1)
            else:
                dst.write(src.read())

    def create_temporary_file_multiple_bands(self, src, data, count, dytpe) -> str:
        """
        Creates a temporary file for multi-band data.

        Parameters
        ----------
        src : rasterio.io.DatasetReader
            Source dataset for metadata.
        data : numpy.ndarray
            Multi-band data to be written.
        count : int
            Number of bands.
        dytpe
            Data Type of the bands.
        Returns
        -------
        str
            Path to the temporary file.
        """
        print(f"data: {type(data)}")
        print(f"src: {type(src)}")

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            self.write_bands_to_file(temp_file.name, src, data, count, dytpe)
            return temp_file.name

    def save_image_multiple_bands(self, src, data, filename, count, dytpe):
        """
        Saves multi-band data to a persistent file.

        Parameters
        ----------
        src : rasterio.io.DatasetReader
            Source dataset for metadata.
        data : numpy.ndarray
            Multi-band data to be written.
        filename : str
            Name of the saved file.

        Returns
        -------
        str
            Path to the saved file.
        """
        file_path = self.get_unique_filename(self._config.download_directory, filename, self._config.selected_file_type)
        self.write_bands_to_file(file_path, src, data, count, dytpe)
        return file_path

    def ndvi_calculation(self, selected_item):
        """
        Calculates NDVI from Red and Near Infrared bands.
        Optionally, plots, downloads and imports the calculated image.

        Parameters
        ----------
        selected_item : pystac.item.Item
            STAC item to be processed.
        """
        band_provider_mapping = {
            "landsat-c2-l2": ("red", "nir08"),
            "landsat-c2-l1": ("red", "nir08"),
            self._provider.EARTH_SEARCH: ("red", "nir"),
            self._provider.PLANETARY_COMPUTER: ("B04", "B08"),
        }
        try:
            red_band, nir_band = band_provider_mapping.get(self._collection) or band_provider_mapping[self._provider]
        except KeyError:
            raise NDVICalculationError("NDVI cannot be calculated for this asset.")

        red_url = selected_item.assets[red_band].href
        nir_url = selected_item.assets[nir_band].href

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
                with rasterio.open(
                        ndvi_path,
                        'w',
                        driver='GTiff',
                        height=red_src.height,
                        width=red_src.width,
                        count=1,
                        dtype='float32',
                        crs=red_src.crs,
                        transform=red_src.transform,
                ) as dst:
                    dst.write(ndvi.astype('float32'), 1)
                if self._config.import_checked:
                    import_into_qgis(ndvi_path, "NDVI " + self._provider.plot_title + "-" + self._collection)
