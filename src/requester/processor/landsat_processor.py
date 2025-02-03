"""
landsat_processor
=================
Defines the LandsatProcessor class for handling Landsat data from Planetary Computer and Earth Search.
"""

from .base_processor import Processor
import os
import rasterio.features
import matplotlib.pyplot as plt
import tempfile
from ...utils import import_into_qgis


class LandsatProcessor(Processor):
    """
    A subclass of Processor.

    Processes Landsat collections (Landsat Collection 2 Level 1 & Landsat Collection 2 Level 1).
    The data can be requested as True Color, False Color, NDVI and single bands.
    Additionally, data can be plotted, downloaded or imported into QGIS.

    Parameters
    ----------
    config : RequestConfig
        Configuration containing user options.
    provider : Provider
        Satellite Data Provider, either PLANETARY_COMPUTER or EARTH_SEARCH.
    collection : str
        Landsat collection, either landsat-c2-l1 or landsat-c2-l2.
    """
    def __init__(self, config, provider, collection):
        super().__init__(config, provider, collection)
        self._config = config
        self._provider = provider
        self._collection = collection

    def process(self, selected_item):
        """
        Implements processing for Landsat collections.

        Parameters
        ----------
        selected_item : pystac.item.Item
            STAC item to be processed.

        Raises
        ------
        ValueError
            If the collection type is unsupported.
        """
        if self._collection not in ["landsat-c2-l1", "landsat-c2-l2"]:
            raise ValueError(f"Unsupported Landsat collection: {self._collection}")

        bands = self.get_bands()

        if self._provider == self._provider.EARTH_SEARCH:
            os.environ["AWS_REQUEST_PAYER"] = "requester"
            os.environ["AWS_PROFILE"] = "default"

        for band in bands:
            if band in ["True Color", "False Color"]:
               self.process_true_and_false_color(selected_item, band)
            else:
                # handle single bands
                self.process_single_band(selected_item, band)

        # NDVI
        if self._config.additional_options and self._config.additional_options.ndvi_checked:
            self.ndvi_calculation(selected_item)

    def get_bands(self):
        """
        Gets selected Bands and/or Color Composition.
        If nothing is selected, the default value gets returned.
        Landsat Collection 2 Level 1 -> False Color.
        Landsat Collection 2 Level 2 -> True Color.

        Returns
        -------
        list of str
            Selected bands or default bands based on the collection.
        """
        default_color_mapping = {
            "landsat-c2-l1": ["False Color"],
            "landsat-c2-l2": ["True Color"],
        }
        if self._config.additional_options and self._config.additional_options.bands:
            return self._config.additional_options.bands
        return default_color_mapping.get(self._collection, [])

    def process_true_and_false_color(self, selected_item, band):
        """
        Processes True Color or False Color compositions.

        Parameters
        ----------
        selected_item : pystac.item.Item
            STAC item to be processed.

        band : str
            The color composition type (True Color or False Color).

        Raises
        ------
        ValueError
            If the color composition type is unsupported.
        """
        if band == "True Color":
            red_name = self._provider.collection_mapping[self._collection]["Red"]
            green_name = self._provider.collection_mapping[self._collection]["Green"]
            blue_name = self._provider.collection_mapping[self._collection]["Blue"]
            self.process_color(selected_item, red_name, green_name, blue_name, band)
        elif band == "False Color":
            nir_name = self._provider.collection_mapping[self._collection]["Near Infrared"]
            red_name = self._provider.collection_mapping[self._collection]["Red"]
            green_name = self._provider.collection_mapping[self._collection]["Green"]
            self.process_color(selected_item, nir_name, red_name, green_name, band)
        else:
            raise ValueError(f"Unsupported Color Composition: {band}")

    def process_color(self, selected_item, red_name, green_name, blue_name, band):
        """
        Processes and optionally plots, downloads or imports color compositions (True Color or False Color).

        Parameters
        ----------
        selected_item : pystac.item.Item
            STAC item to be processed.
        red_name : str
            Asset name for the red band.
        green_name : str
            Asset name for the green band.
        blue_name : str
            Asset name for the blue band.
        band : str
            The color composition type.
        """
        red_url = selected_item.assets[red_name].href
        green_url = selected_item.assets[green_name].href
        blue_url = selected_item.assets[blue_name].href

        color_with_alpha = self.true_false_color_calculation(red_url, green_url, blue_url, True)

        # import
        if self._config.plot_checked:
            plt.figure()
            plt.imshow(color_with_alpha.astype('uint8'))
            plt.title(self._provider.plot_title + " " + self._collection + "-" + band)
            plt.axis("off")
            plt.show()

        # download
        if self._config.download_checked:
            with rasterio.open(red_url) as red_src:
                file_path = self.save_image_multiple_bands(red_src, color_with_alpha,
                                                           f"{self._provider.filename}_{self._collection}_{band.lower().replace(' ', '_')}",
                                                           4, 'uint8')
                # import
                if self._config.import_checked:
                    import_into_qgis(file_path,
                                     self._provider.qgis_layer_name + " " + self._collection + "-" + band)

        # just import no download
        if self._config.import_checked and not self._config.download_checked:
            with rasterio.open(red_url) as red_src:
                filepath = self.create_temporary_file_multiple_bands(red_src, color_with_alpha, 4, 'uint8')
                import_into_qgis(filepath, self._provider.qgis_layer_name + " " + self._collection + "-" + band)

    def process_single_band(self, selected_item, band):
        """
        Processes and optionally plots, downloads or imports a single band.

        Parameters
        ----------
        selected_item : pystac.item.Item
            STAC item to be processed.
        band : str
            The color composition type.
        """
        asset_name = self._provider.collection_mapping[self._collection][band]
        asset_url = selected_item.assets[asset_name].href

        if self._config.plot_checked:
            self.plot_image_single_band(asset_url, band)
        if self._config.download_checked:
            self.save_image_single_band(asset_url, band)
        if self._config.import_checked:
            if self._provider == self._provider.PLANETARY_COMPUTER:
                import_into_qgis(asset_url, band)
            else:
                filepath = self.create_temporary_file_single_band(asset_url)
                import_into_qgis(filepath, self._provider.qgis_layer_name + " " + self._collection + "-" + band)

    def create_temporary_file_single_band(self, asset_url) -> str:
        """
        Creates a temporary file for single-band data.

        Parameters
        ----------
        asset_url : str
            URL of the asset to process.

        Returns
        -------
        str
            Path to the temporary file.
        """
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            with rasterio.open(asset_url) as src:
                self.write_bands_to_file(temp_file.name, src, src.read(1), 1, src.dtypes[0])
            return temp_file.name
