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
import rasterio
import rasterio.mask
from rasterio.features import geometry_mask
import json
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

from threading import Lock
logger = logging.getLogger("SatHubAI.ChangeDetectionProcessor")

class ChangeDetectionProcessor(Processor):
    def __init__(self, config, provider, collection, invekos_manager):
        super().__init__(config, provider, collection)
        self.threshold_ndvi_change = 0.25  # significant NDVI decrease
        self.invekos_manager = invekos_manager
        self.change_results = {}  # store results by field ID
        self.ndvi_results = {}
        self.max_workers =  min(4, os.cpu_count() * 2)
        self.gdal_lock = Lock()
        logger.info(f"Initializing with {self.max_workers} workers")
        self.transformed_geometries = {}  # save transformations

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
                feature = future_to_field[future]
                field_id = feature['id']
                try:
                    field_results = future.result()
                    self.change_results[field_id] = field_results['changes']

                    # log results for this field
                    crop_type = feature['snar_bezeichnung']
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
        self.update_layer_styling(invekos_layer)

    def process_field(self, feature: QgsFeature, items_list: List) -> Dict:
        """
        Process satellite data for a single field in parallel using threads.

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
            ndvi_workers = min(4, len(items_list))
            field_id = feature['id']
            with ThreadPoolExecutor(max_workers=ndvi_workers) as ndvi_executor:
                future_to_date = {
                    ndvi_executor.submit(self.calculate_masked_ndvi, item, feature): item
                    for item in items_list
                }

                for future in as_completed(future_to_date):
                    item = future_to_date[future]
                    try:
                        ndvi_mean = future.result()
                        if ndvi_mean is not None:
                            date = datetime.strptime(
                                item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ'
                            )
                            results['dates'].append(date)
                            results['ndvi_values'].append(ndvi_mean)

                    except Exception as e:
                        logger.error(f"Error calculating NDVI for field {feature['id']}: {str(e)}")

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
                        f"to {min_ndvi:.3f} ({min_date})"
                    )

        except Exception as e:
            logger.error(f"Error in process_field: {str(e)}")
            raise

        # save all calculated NDVI values
        self.ndvi_results[field_id] = results
        return results

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
                QgsRendererCategory("1", change_symbol, "Change Detected"),
                QgsRendererCategory("0", no_change_symbol, "No Change"),
                QgsRendererCategory("-1", error_symbol, "Failed / No Data")
            ]

            # get field IDs for different categories
            # field ids with changes
            change_ids = {str(fid) for fid, changes in self.change_results.items() if changes}

            # field ids with fewer than 2 ndvi values
            error_ids = {
                str(fid) for fid, results in self.ndvi_results.items()
                if not results or 'ndvi_values' not in results or len(results['ndvi_values']) < 2
            }

            # dynamically creat expression
            expression_parts = []
            if change_ids:
                expression_parts.append(f"WHEN id IN ({','.join(change_ids)}) THEN '1'")
            if error_ids:
                expression_parts.append(f"WHEN id IN ({','.join(error_ids)}) THEN '-1'")

            # default is '0' (No Change)
            expression = f"CASE {' '.join(expression_parts)} ELSE '0' END" if expression_parts else "'0'"

            # apply styling
            renderer = QgsCategorizedSymbolRenderer(expression, categories)
            layer.setRenderer(renderer)

            # set tool tips
            layer.setDisplayExpression(display_expression)

            # refresh display
            layer.triggerRepaint()
            QgsProject.instance().reloadAllLayers()

        except Exception as e:
            logger.error(f"Error in update_layer_styling: {str(e)}")
            raise

    def calculate_masked_ndvi(self, selected_item, feature):
        """
        Calculate NDVI for a field using masked satellite data.

        Parameters
        ----------
        selected_item : Item
            STAC item containing satellite data
        feature : QgsFeature
            INVEKOS field feature

        Returns
        -------
        float or None
            Mean NDVI value for the field or None if calculation fails
        """
        field_id = feature['id']
        date = datetime.strptime(selected_item.properties['datetime'], '%Y-%m-%dT%H:%M:%S.%fZ')

        try:
            # transform invekos field from "EPSG:4326" to "EPSG:32633"
            geom = from_geojson(feature.geometry().asJson())
            geom_gdf = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:32633").iloc[0]
            bounds = geom_gdf.bounds

            red_band, nir_band, green_band, blue_band = self.band_mapping.get(self._collection) or self.band_mapping[self._provider]
            red_url = selected_item.assets[red_band].href
            nir_url = selected_item.assets[nir_band].href

            with self.gdal_lock:
                with Env(GDAL_CACHEMAX=512):
                    if field_id not in self.transformed_geometries:
                        with rasterio.open(red_url) as red_src:
                            # Transform geometry
                            satellite_crs = QgsCoordinateReferenceSystem(red_src.crs.to_string())
                            invekos_crs = QgsCoordinateReferenceSystem("EPSG:4326")

                            if not satellite_crs.isValid():
                                logger.error(f"Invalid satellite CRS: {red_src.crs.to_string()}")
                                return None

                            if not invekos_crs.isValid():
                                logger.error(f"Invalid INVEKOS CRS")
                                return None

                            geom = feature.geometry()

                            if not geom:
                                logger.error(f"No geometry found for field {field_id}")
                                return None

                            if satellite_crs != invekos_crs:
                                try:
                                    transform = QgsCoordinateTransform(invekos_crs, satellite_crs, QgsProject.instance())
                                    geom.transform(transform)

                                    # validate transformation
                                    if not geom.isGeosValid():
                                        logger.error(f"Invalid geometry after transform for field {field_id}")
                                        return None

                                    # check if still the same
                                    if feature.geometry().boundingBox() == geom.boundingBox():
                                        logger.error(f"Transform failed for field {field_id}: Coordinates unchanged")
                                        return None

                                except Exception as e:
                                    logger.error(f"Transform failed for field {field_id}: {str(e)}")
                                    return None

                            # get bounds
                            bounds = geom.boundingBox()

                            # save in cache
                            self.transformed_geometries[field_id] = geom
                    else:
                            geom = self.transformed_geometries[field_id]
                            bounds = geom.boundingBox()

                with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                        # get window
                        window = red_src.window(
                            bounds.xMinimum(),
                            bounds.yMinimum(),
                            bounds.xMaximum(),
                            bounds.yMaximum()
                        )

                        # check if window is valid
                        if (window.col_off >= red_src.width or
                                window.row_off >= red_src.height or
                                window.col_off + window.width <= 0 or
                                window.row_off + window.height <= 0):
                            logger.debug(f"Field {field_id} outside image bounds")
                            return None

                        # read data using integer window coordinates
                        window = rasterio.windows.Window(
                            col_off=int(window.col_off),
                            row_off=int(window.row_off),
                            width=int(window.width),
                            height=int(window.height)
                        )

                        # read both bands
                        red = red_src.read(1, window=window).astype(float)
                        nir = nir_src.read(1, window=window).astype(float)

                        if red.size == 0 or nir.size == 0:
                            logger.debug(f"Empty data read for field {field_id} at {date}")
                            return None

                        # create mask
                        geom_json = json.loads(geom.asJson())
                        mask = geometry_mask(
                            [geom_json],
                            out_shape=(window.height, window.width),
                            transform=red_src.window_transform(window),
                            invert=True
                        )

                        # calculate NDVI only for valid pixels
                        valid_pixels = mask & (red != 0) & (nir != 0)
                        if not np.any(valid_pixels):
                            return None

                        ndvi = np.zeros_like(red)
                        ndvi[valid_pixels] = (nir[valid_pixels] - red[valid_pixels]) / (
                                    nir[valid_pixels] + red[valid_pixels] + 1e-10)
                        ndvi = np.clip(ndvi, -1, 1)

                        return float(np.mean(ndvi[valid_pixels]))

        except Exception as e:
            logger.error(f"Error processing field {field_id}: {str(e)}")
            logger.exception("Full traceback:")
            return None