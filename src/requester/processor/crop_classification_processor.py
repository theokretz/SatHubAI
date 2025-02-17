"""
crop_classification_processor.py
=================
Processor for classifying crops as either single crops (Reinsaat) or mixed crops (Mischsaat) using satellite data features.
"""
import joblib
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from qgis.core import QgsFillSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer, QgsProject
from qgis.core import QgsField, edit
from PyQt5.QtCore import QVariant
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from shapely.io import from_geojson
from pathlib import Path
from ...crop_classification.crop_types import is_single_crop
import pandas as pd
import geopandas as gpd
from .base_processor import Processor
from scipy.stats import kurtosis, skew

logger = logging.getLogger("SatHubAI.CropClassificationProcessor")


class CropClassificationProcessor(Processor):
    """
       Processor for classifying crops as single (Reinsaat) or mixed (Mischsaat) using satellite data.
    """
    def __init__(self, config, provider, collection, invekos_manager):
        super().__init__(config, provider, collection)
        self.invekos_manager = invekos_manager
        self.max_workers = min(4, os.cpu_count() * 2)
        self.transformed_geometries = {}
        self.feature_results = {}
        self.model = self.load_model()
        self.prediction_results = {}
        self.invalid_fields = []

    @staticmethod
    def load_model():
        """
        Load the pre-trained Random Forest model.

        Returns
        -------
        object or None
            Loaded model if successful, otherwise None.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))  # `processor`directory
        model_path = os.path.join(base_path, "..", "..", "crop_classification", "random_forest_model.pkl")
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error while loading the model: {e}")
            return None

    def process(self, time_series_items: List) -> None:
        """
        Process a time series of satellite images to extract features and classify crops.

        Parameters
        ----------
        time_series_items : list
            List of satellite image items (STAC items) to analyze.
        """
        invekos_layer = self.invekos_manager.get_current_layer()
        if not invekos_layer or not invekos_layer.isValid():
            logger.error("No valid INVEKOS layer found")
            return

        # sort items by date
        items_list = list(time_series_items)
        items_list.sort(
            key=lambda x: datetime.strptime(
                x.properties['datetime'],
                '%Y-%m-%dT%H:%M:%S.%fZ'
            )
        )
        logger.info(f"Processing {len(items_list)} images with {self.max_workers} workers")

        # process fields in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures for each field
            future_to_field = {
                executor.submit(
                    self.extract_field_features,
                    feature,
                    items_list
                ): feature
                for feature in invekos_layer.getFeatures()
            }

            # process results as they complete
            for future in as_completed(future_to_field):
                field = future_to_field[future]
                field_id = field['id']
                crop_label = is_single_crop(field['snar_bezeichnung'])
                area = field['sl_flaeche_brutto_ha']
                try:
                    field_features = future.result()
                    # only if features were extracted
                    if field_features:
                        self.feature_results[field_id] = {
                        'field_id': field_id,
                        'label': crop_label,
                        'area': area
                        }
                        self.feature_results[field_id].update(field_features)
                except Exception as e:
                    logger.error(f"Error processing field {field_id}: {str(e)}yippie")

        self.predict_crop_types()
        self.assign_predictions_to_layer(invekos_layer)
        self.update_layer_styling(invekos_layer)
        logger.info(f"Successfully predicted crop types of {len(self.prediction_results)} crops")

        # save extracted features and prediction
        if self._config.download_checked:
            self.save_features()

    def predict_crop_types(self):
        """
        Predict crop types using the trained Random Forest model.

        Stores predictions in self.prediction_results.
        """
        if self.model is None:
            logger.error("The model is not loaded.")

        if self.feature_results:
            features_df = pd.DataFrame.from_dict(self.feature_results, orient='index').drop(columns=["field_id", "label"])
            predictions = self.model.predict(features_df.values)

            # store predictions in `self.prediction_results`
            self.prediction_results = {
                field["field_id"]: {**field, "crop_prediction": int(pred)}
                for field, pred in zip(self.feature_results.values(), predictions)
            }
        else:
            logger.error("No features found!")

    def filter_images_for_field(self, items_list, field, field_bounds):
        """
        Filter images based on bounding box overlap with invekos field.

        Parameters
        ----------
        items_list : list
            List of satellite images (STAC items).
        field : QgsFeature
            The INVEKOS field
        field_bounds : tuple
            Bounding box of the field in UTM coordinates.

        Returns
        -------
        list
            List of relevant images that intersect the field.
        """
        relevant_images = []

        try:
            # create box for field
            field_box = box(*field_bounds)

            for item in items_list:
                # create box for image
                image_box = box(*self.transform_bounds_to_utm(item.bbox))

                # check intersection
                if field_box.intersects(image_box):
                    relevant_images.append(item)

        except Exception as e:
            logger.error(f"Error in filtering for field {field['id']}: {str(e)}")
            return items_list  # return all images if filtering fails

        return relevant_images

    def transform_bounds_to_utm(self, bounds):
        """
        Transform bounding box coordinates from WGS84 to UTM33N.

        Parameters
        ----------
        bounds : tuple
            Bounding box in EPSG:4326.

        Returns
        -------
        tuple
            Bounding box in EPSG:32633.
        """
        box_geom = box(*bounds)
        transformed = gpd.GeoSeries([box_geom], crs="EPSG:4326").to_crs("EPSG:32633").iloc[0]
        return transformed.bounds

    def extract_field_features(self, field, items_list: List) -> Dict:
        """
        Extract vegetation indices from satellite images for a given field.

        Parameters
        ----------
        field : QgsFeature
            The INVEKOS field
        items_list : List
            List of satellite images (STAC items).

        Returns
        -------
        dict
            Dictionary containing extracted features per field
        """
        results = {
            'ndvi_values': [],
            'evi_values': [],
            'savi_values': [],
            'msavi_values': [],
            'ndwi_values': [],
            'gci_values' : [],
            'dates': []
        }
        try:
            field_id = field['id']
            # transform invekos field from "EPSG:4326" to "EPSG:32633"
            geom = from_geojson(field.geometry().asJson())
            geom_gdf = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:32633").iloc[0]
            bounds = geom_gdf.bounds

            # filter images
            relevant_images = self.filter_images_for_field(items_list, field, bounds)

            for item in relevant_images:
                if field_id in self.invalid_fields:
                    continue
                try:
                    red_band, nir_band, green_band, blue_band = self.band_mapping.get(self._collection) or self.band_mapping[self._provider]
                    red_url = item.assets[red_band].href
                    nir_url = item.assets[nir_band].href
                    green_url = item.assets[green_band].href
                    blue_url = item.assets[blue_band].href

                    with rasterio.Env(GDAL_CACHEMAX=512):
                        with rasterio.open(red_url) as red_src:
                            # get window and transform
                            window = red_src.window(*bounds)
                            if not self.is_valid_window(window, red_src.width, red_src.height):
                                logger.warning(f"Invalid window for field {field_id}")
                                self.invalid_fields.append(field_id)
                                continue

                            window = self.round_window(window)
                            window_transform = red_src.window_transform(window)

                            # read data
                            red = red_src.read(1, window=window).astype(float)

                            with rasterio.open(nir_url) as nir_src:
                                nir = nir_src.read(1, window=window).astype(float)

                            if red.size == 0 or nir.size == 0:
                                logger.warning(f"Skipping field {field_id} - Field likely too small: {field['sl_flaeche_brutto_ha']}")
                                self.invalid_fields.append(field_id)
                                continue

                            if red.shape != nir.shape:
                                logger.warning(f"Shape mismatch for field {field_id}: red={red.shape}, nir={nir.shape}")
                                self.invalid_fields.append(field_id)
                                continue

                            # create mask
                            mask = geometry_mask(
                                [geom_gdf],
                                out_shape=(window.height, window.width),
                                transform=window_transform,
                                invert=True
                            )

                            # validate mask shape
                            if mask.shape != red.shape:
                                logger.warning(f"Mask shape mismatch for field {field['id']}: mask={mask.shape}, data={red.shape}")
                                continue

                            # calculate indices only for valid pixels
                            valid_pixels = mask & (red > -1) & (nir > -1)
                            if not np.any(valid_pixels):
                                logger.warning(f"No valid pixels for field {field['id']}")
                                continue

                            # initialize arrays for indices
                            ndvi = np.zeros_like(red)
                            evi = np.zeros_like(red)
                            savi = np.zeros_like(red)
                            msavi = np.zeros_like(red)
                            ndwi = np.zeros_like(red)
                            gci = np.zeros_like(red)


                            # calculate NDVI
                            ndvi[valid_pixels] = (nir[valid_pixels] - red[valid_pixels]) / (
                                    nir[valid_pixels] + red[valid_pixels])
                            ndvi = np.clip(ndvi, -1, 1)
                            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

                            # calculate EVI if blue band is available
                            if blue_url:
                                with rasterio.open(blue_url) as blue_src:
                                    blue = blue_src.read(1, window=window).astype(float)
                                    valid_pixels_evi = valid_pixels & (blue > -1)
                                    evi[valid_pixels_evi] = 2.5 * (
                                            (nir[valid_pixels_evi] - red[valid_pixels_evi]) /
                                            (nir[valid_pixels_evi] + 6 * red[valid_pixels_evi] - 7.5 * blue[valid_pixels_evi] + 1)
                                    )
                                    evi = np.nan_to_num(evi, nan=0.0, posinf=1.0, neginf=-1.0)

                            # calculate SAVI
                            L = 0.5  # soil brightness correction factor
                            savi[valid_pixels] = (
                                                (nir[valid_pixels] - red[valid_pixels]) /
                                                (nir[valid_pixels] + red[valid_pixels] + L)) * (1 + L)
                            savi = np.clip(savi, -1, 1)
                            savi = np.nan_to_num(savi, nan=0.0, posinf=1.0, neginf=-1.0)

                            # Calculate MSAVI
                            msavi[valid_pixels] = (2 * nir[valid_pixels] + 1 - np.sqrt((2 * nir[valid_pixels] + 1)
                                                ** 2 - 8 * (nir[valid_pixels] - red[valid_pixels]))) / 2
                            msavi = np.nan_to_num(msavi, nan=0.0, posinf=1.0, neginf=-1.0)

                            # calculate NDWI and GCI if green band is available
                            if green_url:
                                with rasterio.open(green_url) as green_src:
                                    green = green_src.read(1, window=window).astype(float)
                                    valid_pixels_ndwi = valid_pixels & (green > -1)
                                    ndwi[valid_pixels_ndwi] = (
                                            (green[valid_pixels_ndwi] - nir[valid_pixels_ndwi]) /
                                            (green[valid_pixels_ndwi] + nir[valid_pixels_ndwi] + 1e-10)
                                    )
                                    ndwi = np.clip(ndwi, -1, 1)
                                    ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=1.0, neginf=-1.0)

                                    valid_pixels_gci = valid_pixels & (green > -1)
                                    gci[valid_pixels_gci] = (nir[valid_pixels_gci] / green[valid_pixels_gci]) - 1
                                    gci = np.nan_to_num(gci, nan=0.0, posinf=1.0, neginf=-1.0)

                            # store results
                            date = datetime.strptime(item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ')
                            results['dates'].append(date)
                            results['ndvi_values'].append(float(np.mean(ndvi[valid_pixels])))

                            if blue is not None:
                                results['evi_values'].append(float(np.mean(evi[valid_pixels_evi])))

                            results['savi_values'].append(float(np.mean(savi[valid_pixels])))
                            results['msavi_values'].append(float(np.mean(msavi[valid_pixels])))

                            if green is not None:
                                results['ndwi_values'].append(float(np.mean(ndwi[valid_pixels_ndwi])))
                                results['gci_values'].append(float(np.mean(gci[valid_pixels_gci])))

                except Exception as e:
                    logger.error(f"Error processing image for field {field['id']}: {str(e)}")
                    continue

            return self.calculate_statistics(results)

        except Exception as e:
            logger.error(f"Error processing field {field['id']}: {str(e)}")
            return None


    def calculate_statistics(self, results):
        """
        Compute statistical metrics for extracted vegetation indices.

        Parameters
        ----------
        results : dict
            Dictionary containing vegetation index values over time.

        Returns
        -------
        dict
            Statistical summary of vegetation indices.
        """
        stats = {}
        if len(results['dates']) > 0:
         for index_name in ['ndvi', 'evi', 'savi', 'msavi', 'ndwi', 'gci']:
             values = results.get(f'{index_name}_values', [])
             if values:
                 stats.update({
                     f'{index_name}_mean': np.mean(values),
                     f'{index_name}_std': np.std(values),
                     f'{index_name}_min': np.min(values),
                     f'{index_name}_max': np.max(values),
                     f'{index_name}_range': np.ptp(values),
                     f'{index_name}_median': np.median(values),
                 })

                 # calculate trend if more than one observation
                 if len(values) > 1:
                     days = [(date - results['dates'][0]).days for date in results['dates']]
                     try:
                         stats[f'{index_name}_trend'] = np.polyfit(days, values, 1)[0]
                     except Exception as e:
                         stats[f'{index_name}_trend'] = None

                     peak_idx = np.argmax(values)
                     stats[f'{index_name}_time_to_peak'] = days[peak_idx]

                     # rate of change features
                     changes = np.diff(values)
                     stats[f'{index_name}_max_increase'] = np.max(changes)
                     stats[f'{index_name}_max_decrease'] = np.min(changes)
                     stats[f'{index_name}_mean_change'] = np.mean(changes)

                     stats[f'{index_name}_cv'] = stats[f'{index_name}_std'] / stats[f'{index_name}_mean']

                 if len(results['dates']) >= 3:  # These metrics need at least 3 points
                     stats[f'{index_name}_skewness'] = float(skew(values))
                     stats[f'{index_name}_kurtosis'] = float(kurtosis(values))
                 else:
                     stats[f'{index_name}_skewness'] = None
                     stats[f'{index_name}_kurtosis'] = None

                 # growing season features
                 if len(results['dates']) >= 3:  # Need at least 3 points for season divisions
                     stats[f'{index_name}_growing_season'] = days[-1]
                     stats[f'{index_name}_early_season_mean'] = np.mean(values[:len(values) // 3])
                     stats[f'{index_name}_mid_season_mean'] = np.mean(
                         values[len(values) // 3:2 * len(values) // 3])
                     stats[f'{index_name}_late_season_mean'] = np.mean(values[2 * len(values) // 3:])

         stats['n_observations'] = len(results['dates'])
         return stats

    def is_valid_window(self, window, width, height):
        """
        Check if a rasterio window is valid.

        Parameters
        ----------
        window : rasterio.windows.Window
           Window object to check.
        width : int
           Raster width.
        height : int
           Raster height.

        Returns
        -------
        bool
           True if the window is valid, False otherwise.
        """
        return not (window.col_off >= width or
                    window.row_off >= height or
                    window.col_off + window.width <= 0 or
                    window.row_off + window.height <= 0)

    def round_window(self, window):
        """
        Round window coordinates to integer values.

        Parameters
        ----------
        window : rasterio.windows.Window
           Window object.

        Returns
        -------
        rasterio.windows.Window
           Window with integer coordinates.
        """
        return rasterio.windows.Window(
            col_off=int(window.col_off),
            row_off=int(window.row_off),
            width=int(window.width),
            height=int(window.height)
        )
    def save_features(self):
        """
        Save extracted features and predictions to a CSV file.
        """
        try:
            if self.prediction_results:
                # convert features dictionary to DataFrame
                features_df = pd.DataFrame.from_dict(self.prediction_results, orient='index')

                # convert 1 to Single Crop and 0 to Mixed Crop
                if "crop_prediction" in features_df.columns:
                    features_df["crop_prediction"] = features_df["crop_prediction"].map({1: "Single Crop", 0: "Mixed Crop"})

                # reorder columns move crop_prediction after field_id
                if "crop_prediction" in features_df.columns:
                    cols = ["field_id", "crop_prediction"] + [col for col in features_df.columns if
                                                              col not in ["field_id", "crop_prediction"]]
                    features_df = features_df[cols]

                # create output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    self._config.download_directory,
                    f'crop_classification_predictions_{timestamp}.csv'
                )

                # save to CSV
                features_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(self.prediction_results)} features to {output_path}")

            else:
                logger.warning("No features to save.")
                return
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

    def update_layer_styling(self, layer):
        """
        Updates the QGIS layer styling based on crop classification results.

        Colors:
        - Blue for "Single Crop"
        - Yellow for "Mixed Crop"

        Parameters
        ----------
        layer : QgsVectorLayer
            The INVEKOS vector layer to style
        """
        try:
            if not layer or not layer.isValid():
                logger.error("Invalid layer for styling.")
                return

            # define new colors for classifications
            single_crop_symbol = QgsFillSymbol.createSimple({'color': '#0000FF'})  # Blue for Single Crop
            mixed_crop_symbol = QgsFillSymbol.createSimple({'color': '#FFD700'})  # Yellow for Mixed Crop
            unknown_symbol = QgsFillSymbol.createSimple({'color': '#808080'})  # Gray for Unknown

            # define categories based on crop_prediction
            categories = [
                QgsRendererCategory("Single Crop", single_crop_symbol, "Single Crop"),
                QgsRendererCategory("Mixed Crop", mixed_crop_symbol, "Mixed Crop"),
                QgsRendererCategory(None, unknown_symbol, "Unknown")  # Handles NULL values
            ]

            # create categorized renderer using the crop_prediction field
            renderer = QgsCategorizedSymbolRenderer("crop_prediction", categories)
            layer.setRenderer(renderer)

            # refresh layer in QGIS
            layer.triggerRepaint()
            QgsProject.instance().reloadAllLayers()

        except Exception as e:
            logger.error(f"Error updating layer styling: {e}")
            return

    def assign_predictions_to_layer(self, layer):
        """
        Assign predicted crop classification to QGIS features.

        Parameters
        ----------
        layer : QgsVectorLayer
            INVEKOS vector layer to update.
        """
        if not self.prediction_results:
            logger.warning("No predictions to assign.")
            return

        # add crop_prediction field to layers before assigning values
        if layer.fields().indexFromName("crop_prediction") == -1:
            layer.dataProvider().addAttributes([QgsField("crop_prediction", QVariant.String)])
            layer.updateFields()

        # add predictions
        with edit(layer):
            for feature in layer.getFeatures():
                field_id = feature["id"]
                if field_id in self.prediction_results:
                    pred_value = self.prediction_results[field_id]["crop_prediction"]
                    feature["crop_prediction"] = "Single Crop" if pred_value == 1 else "Mixed Crop"
                    layer.updateFeature(feature)
