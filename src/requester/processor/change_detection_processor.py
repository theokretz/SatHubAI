"""
change_detection_processor.py
Change detection processor using parallel processing.
"""
import numpy as np
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime

import pandas as pd
import rasterio
import rasterio.mask
from rasterio.features import geometry_mask
import json
from qgis.core import QgsField, edit
from PyQt5.QtCore import QVariant
from shapely.io import from_geojson
import geopandas as gpd
from .processor import Processor
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsSymbol,
    QgsFeatureRenderer,
    QgsRuleBasedRenderer,
    QgsFillSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsFeature,
    QgsGeometry,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer
)

from rasterio.env import Env

logger = logging.getLogger("SatHubAI.ChangeDetectionProcessor")

class ChangeDetectionProcessor(Processor):
    def __init__(self, config, provider, collection, invekos_manager):
        super().__init__(config, provider, collection)
        self.threshold_ndvi_change = 0.25  # significant NDVI decrease
        self.invekos_manager = invekos_manager
        self.ndvi_results = {}
        self.max_workers =  min(4, os.cpu_count() * 2)
        logger.info(f"Initializing with {self.max_workers} workers")
        self.transformed_geometries = {}  # save transformations
        self.invalid_fields = []

    def process(self, time_series_items):
        """
        Process a time series of satellite images to detect NDVI changes.

        Parameters
        ----------
        time_series_items : list of STAC items
            List of satellite image items to analyze, each containing timestamp and band data.

        Returns
        -------
        None
            Updates internal change_results and ndvi_results, and modifies INVEKOS layer styling.
        """
        invekos_layer = self.invekos_manager.get_current_layer()
        if not invekos_layer or not invekos_layer.isValid():
            logger.error("No valid INVEKOS layer found")
            return

        # convert to list and sort by date
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
                    self.process_field,
                    feature,
                    items_list
                ): feature
                for feature in invekos_layer.getFeatures()
            }

            # process results as they complete
            for future in as_completed(future_to_field):
                field = future_to_field[future]
                field_id = field['id']
                try:
                    field_results = future.result()
                    # store results
                    self.ndvi_results[field_id] = field_results

                    # log results for this field
                    crop_type = field['snar_bezeichnung']
                    self.ndvi_results[field_id]['crop_type'] = crop_type
                    logger.info(f"\nProcessed Field {field_id} - Crop Type: {crop_type}")
                    if field_results['dates']:
                        for date, ndvi in zip(field_results['dates'], field_results['ndvi_values']):
                            logger.debug(f"Date: {date.strftime('%Y-%m-%d')}, NDVI: {ndvi:.3f}")

                        if field_results['changes']:
                            logger.info(f"Changes detected for field {field_id} at:")
                            for change_date in field_results['changes']:
                                logger.info(f"- {change_date.strftime('%Y-%m-%d')}")
                    else:
                        logger.warning(f"No valid observations for field {field_id}")

                except Exception as e:
                    logger.error(f"Error processing field {field_id}: {str(e)}")

        # update layer styling based on results
        self.assign_change_detection_to_layer(invekos_layer)
        self.update_layer_styling(invekos_layer)

        # download results
        if self._config.download_checked:
            self.save_change_results()

    def process_field(self, feature: QgsFeature, items_list: List) -> Dict:
        """
        Process satellite data for a single field.

        Parameters
        ----------
        feature : QgsFeature
            The INVEKOS field feature to process.
        items_list : List
            List of time series items to analyze.

        Returns
        -------
        Dict
            Dictionary containing dates, NDVI values, and detected changes.
        """
        results = {'dates': [], 'ndvi_values': [], 'changes': []}

        try:
            field_id = feature['id']

            for item in items_list:
                ndvi_mean = self.calculate_masked_ndvi(item, feature)
                if ndvi_mean is not None:
                    date = datetime.strptime(
                        item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ'
                    )
                    results['dates'].append(date)
                    results['ndvi_values'].append(ndvi_mean)
                else:
                    logger.warning(f"Could not calculate NDVI for field {feature['id']}")

            # detect significant NDVI drops
            if len(results['ndvi_values']) > 1:
                max_ndvi = max(results['ndvi_values'])
                min_ndvi = min(results['ndvi_values'])
                max_date = results['dates'][results['ndvi_values'].index(max_ndvi)]
                min_date = results['dates'][results['ndvi_values'].index(min_ndvi)]

                ndvi_change = max_ndvi - min_ndvi

                if ndvi_change > self.threshold_ndvi_change and (min_date > max_date):
                    results['changes'].append(max_date)
                    logger.info(
                        f"Change detected: NDVI drop from {max_ndvi:.3f} ({max_date}) "
                        f"to {min_ndvi:.3f} ({min_date}) for field {field_id}"
                    )

        except Exception as e:
            logger.error(f"Error in process_field: {str(e)}")
            raise

        return results

    def assign_change_detection_to_layer(self, layer: QgsVectorLayer):
        """
        Assign change detection results to QGIS features.

        Parameters
        ----------
        layer : QgsVectorLayer
            INVEKOS vector layer to update.
        """
        if not self.ndvi_results:
            logger.warning("No NDVI results found")
            return

        # add change_detected to layers before assigning values
        if layer.fields().indexFromName("change_detected") == -1:
            layer.dataProvider().addAttributes([QgsField("change_detected", QVariant.String)])
            layer.updateFields()

        with edit(layer):
            for feature in layer.getFeatures():
                field_id = feature['id']
                feature["change_detected"] = "Unknown"

                if field_id in self.ndvi_results and len(self.ndvi_results[field_id]['ndvi_values']) >= 2:
                    if self.ndvi_results[field_id]['changes']:
                        change_value = "Change Detected"
                    else:
                        change_value = "No Change"

                feature["change_detected"] = change_value
                layer.updateFeature(feature)


    def update_layer_styling(self, layer: QgsVectorLayer):
        """
        Update QGIS layer styling based on change detection results.
        Green - Change detected.
        Red - No change detected.
        Gray - Calculation failed or no data.

        Parameters
        ----------
        layer : QgsVectorLayer
            INVEKOS vector layer to style

        Returns
        -------
        None
            Updates layer styling in QGIS
        """
        try:
            # save tool tips
            display_expression = layer.displayExpression()

            # define symbols
            change_symbol = QgsFillSymbol.createSimple({'color': '#00FF00'})  # green (change detected)
            no_change_symbol = QgsFillSymbol.createSimple({'color': '#FF0000'})  # red (no change)
            error_symbol = QgsFillSymbol.createSimple({'color': '#888888'})  # gray (failed calculation / no data)

            # define categories
            categories = [
                QgsRendererCategory("Change Detected", change_symbol, "Change Detected"),
                QgsRendererCategory("No Change", no_change_symbol, "No Change"),
                QgsRendererCategory("Unknown", error_symbol, "Unknown")
            ]

            # create categorized renderer using the change_detected field
            renderer = QgsCategorizedSymbolRenderer("change_detected", categories)
            layer.setRenderer(renderer)

            # set tool tips
            layer.setDisplayExpression(display_expression)

            # refresh layer in QGIS
            layer.triggerRepaint()
            QgsProject.instance().reloadAllLayers()
        except Exception as e:
            logger.error(f"Error in update_layer_styling: {str(e)}")
            raise

    def calculate_masked_ndvi(self, selected_item, field):
        """
        Calculate NDVI for a field using masked satellite data.

        Parameters
        ----------
        selected_item : Item
            STAC item containing satellite data
        field : QgsFeature
            INVEKOS field feature

        Returns
        -------
        float or None
            Mean NDVI value for the field or None if calculation fails
        """
        field_id = field['id']
        date = datetime.strptime(selected_item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ')

        try:
            if field_id not in self.transformed_geometries:
                # transform invekos field from "EPSG:4326" to "EPSG:32633"
                geom_json = from_geojson(field.geometry().asJson())
                geom = gpd.GeoSeries([geom_json], crs="EPSG:4326").to_crs("EPSG:32633").iloc[0]
                bounds = geom.bounds
                # save in cache
                self.transformed_geometries[field_id] = geom
            else:
                geom = self.transformed_geometries[field_id]
                bounds = geom.bounds

            red_band, nir_band, green_band, blue_band = self.band_mapping.get(self._collection) or self.band_mapping[self._provider]
            red_url = selected_item.assets[red_band].href
            nir_url = selected_item.assets[nir_band].href


            with Env(GDAL_CACHEMAX=512):
                with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                    # get window and transform
                    window = red_src.window(*bounds)
                    if not self.is_valid_window(window, red_src.width, red_src.height):
                        logger.warning(f"Invalid window for field {field_id}")
                        self.invalid_fields.append(field_id)
                        return

                    # rounding because rasterio needs integer coordinates
                    window = self.round_window(window)
                    window_transform = red_src.window_transform(window)

                    # read data
                    red = red_src.read(1, window=window).astype(float)
                    nir = nir_src.read(1, window=window).astype(float)

                    # validate
                    if red.size == 0 or nir.size == 0:
                        logger.warning(f"Skipping field {field_id} - Field likely too small: {field['sl_flaeche_brutto_ha']}")
                        self.invalid_fields.append(field_id)
                        return
                    if red.shape != nir.shape:
                        logger.warning(f"Shape mismatch for field {field_id}: red={red.shape}, nir={nir.shape}")
                        self.invalid_fields.append(field_id)
                        return

                    # create mask
                    mask = geometry_mask(
                        [geom],
                        out_shape=(window.height, window.width),
                        transform=window_transform,
                        invert=True
                    )

                    # validate mask shape
                    if mask.shape != red.shape:
                        logger.warning(f"Mask shape mismatch for field {field['id']}: mask={mask.shape}, data={red.shape}")
                        return

                    # calculate indices only for valid pixels
                    valid_pixels = mask & (red != 0) & (nir != 0)
                    if not np.any(valid_pixels):
                        logger.warning(f"No valid pixels for field {field['id']}")
                        return

                    # initialize arrays for indix
                    ndvi = np.zeros_like(red)

                    #calculate ndvi
                    ndvi[valid_pixels] = (nir[valid_pixels] - red[valid_pixels]) / (
                                nir[valid_pixels] + red[valid_pixels] + 1e-10)
                    ndvi = np.clip(ndvi, -1, 1)

                    return float(np.mean(ndvi[valid_pixels]))
        except Exception as e:
            logger.error(f"Error processing field {field_id}: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def save_change_results(self):
        """
          Save change detection results to a CSV file.

          Parameters:
          -----------
          detection_results : dict
              Dictionary with field IDs as keys and detection results as values
          """
        rows = []

        for field_id, result in self.ndvi_results.items():
            # get change information
            dates = result.get('dates', [])
            ndvi_values = result.get('ndvi_values', [])
            changes = result.get('changes', [])
            crop_type = result.get('crop_type', 'Unknown')

            # format dates for output
            dates_str = [d.strftime('%Y-%m-%d') for d in dates]
            changes_str = [d.strftime('%Y-%m-%d') for d in changes]

            # create a row with all field information
            row = {
                'field_id': field_id,
                'change_detected': 'Yes' if changes else 'No',
                'crop_type': crop_type,
                'change_dates': '; '.join(changes_str),
                'observation_count': len(dates),
                'observation_dates': '; '.join(dates_str),
                'ndvi_values': '; '.join([str(round(val, 4)) for val in ndvi_values])
            }

            # calculate min and max ndvi values
            if ndvi_values:
                min_ndvi = min(ndvi_values)
                max_ndvi = max(ndvi_values)

                # Find the dates for min and max NDVI
                min_ndvi_index = ndvi_values.index(min_ndvi)
                max_ndvi_index = ndvi_values.index(max_ndvi)

                min_ndvi_date = dates_str[min_ndvi_index]
                max_ndvi_date = dates_str[max_ndvi_index]

                row['min_ndvi'] = round(min_ndvi, 4)
                row['min_ndvi_date'] = min_ndvi_date
                row['max_ndvi'] = round(max_ndvi, 4)
                row['max_ndvi_date'] = max_ndvi_date
                row['ndvi_range'] = round(max_ndvi - min_ndvi, 4)

            rows.append(row)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self._config.download_directory,
            f'change_detection_results_{timestamp}.csv'
        )
        # create dataframe and save as CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return df