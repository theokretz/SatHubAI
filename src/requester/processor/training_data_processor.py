"""
training_data_processor.py
=================
Processor for preparing training data from GeoPackage files for the crop classification model.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from scipy.stats import kurtosis, skew
from scipy import stats
from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
import json
import rasterio
from rasterio import Env
from rasterio.features import geometry_mask
from rasterio.mask import mask
from shapely.geometry import box
from .base_processor import Processor
from ...crop_classification.crop_types import is_pure_crop

logger = logging.getLogger("SatHubAI.TrainingDataProcessor")

class TrainingDataProcessor(Processor):
    """
    Processor for preparing training data from Invekos GeoPackage file.
    """

    def __init__(self, config, provider, collection):
        super().__init__(config, provider, collection)
        self.data = None
        self.features = {}
        self.max_workers = min(32, os.cpu_count() * 2)

    def process(self, items):
        try:
            logger.info("Loading data...")
            self.load_data()

            # Sort items by date
            items_list = list(items)
            items_list.sort(
                key=lambda x: datetime.strptime(
                    x.properties['datetime'],
                    '%Y-%m-%dT%H:%M:%S.%fZ'
                )
            )
            logger.info(f"Processing {len(items_list)} images with {self.max_workers} workers")
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_field = {
                        executor.submit(
                            self._process_single_field,
                            row,
                            items_list
                        ): idx
                        for idx, row in self.data.iterrows()
                    }

                    for future in future_to_field:
                        try:
                            idx = future_to_field[future]
                            result = future.result()
                            if result:
                                self.features[idx] = result
                        except Exception as e:
                            logger.error(f"Error processing field {idx}: {str(e)}")

            except Exception as e:
                logger.error(f"Error in feature extraction: {str(e)}")
                raise
            self.save_features()
        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            raise

    def load_data(self):
        """
        Load data from the GeoPackage file and perform initial processing.
        """
        try:
            geopackage_path = Path(r"/src/crop_classification\INSPIRE_SCHLAEGE_2024_BALANCED.gpkg")
            self.data = gpd.read_file(geopackage_path)
            logger.info(f"Loaded {len(self.data)} records from GeoPackage")
            gdf = gpd.read_file(geopackage_path)

            # add classification labels
            self.data['crop_label'] = self.data['SNAR_BEZEICHNUNG'].apply(is_pure_crop)

            # filter out invalid/unknown crops
            valid_mask = (self.data['crop_label'] != -1) & (self.data['crop_label'].notna())
            self.data = self.data[valid_mask].copy()

            logger.info(f"Retained {len(self.data)} valid records after filtering")
        except Exception as e:
            logger.error(f"Error loading GeoPackage: {str(e)}")
            raise

    def _process_single_field(self, feature, items_list):
        """
        Process a single field to extract features.
        """
        try:
            field_features = self.extract_field_features(feature, items_list)

            if field_features:
                features = {
                    'field_id': str(feature['GEO_ID']),
                    'label': int(feature['crop_label']),
                    'area': float(feature['SL_FLAECHE_BRUTTO_HA']),
                }

                features.update(field_features)
                logger.info(f"features for {str(feature['GEO_ID'])} successfully extracted")
                return features
            return None

        except Exception as e:
            logger.error(f"Error processing field {feature['GEO_ID']}: {str(e)}")
            return None

    def filter_images_for_field(self, items_list, feature, field_bounds):
        """
        Filter relevant images based on bounding box overlap
        """
        relevant_images = []

        try:
            # Create shapely box for field
            field_box = box(*field_bounds)

            for item in items_list:
                # Create shapely box for image
                image_box = box(*self.transform_bounds_to_utm(item.bbox))

                # Check intersection
                if field_box.intersects(image_box):
                    relevant_images.append(item)

        except Exception as e:
            print(f"Error in filtering for field {feature['GEO_ID']}: {str(e)}")
            return items_list  # Return all images if filtering fails

        return relevant_images

    def transform_bounds_to_utm(self, bounds):
        """Transform bounds from WGS84 to UTM33N"""
        box_geom = box(*bounds)
        transformed = gpd.GeoSeries([box_geom], crs="EPSG:4326").to_crs("EPSG:32633").iloc[0]
        return transformed.bounds

    def extract_field_features(self, feature, items_list):
        """
        Calculate multiple vegetation indices for a field
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
            # Transform geometry once
            geom = gpd.GeoSeries([feature.geometry], crs="EPSG:31287").to_crs("EPSG:32633").iloc[0]
            bounds = geom.bounds

            # Filter relevant images
            relevant_images = self.filter_images_for_field(items_list, feature, bounds)
            logger.info(f"Field {feature['GEO_ID']}: Processing {len(relevant_images)} relevant images")

            for item in relevant_images:
                try:
                    red_band, nir_band, green_band, blue_band = self.band_provider_mapping.get(self._collection) or self.band_provider_mapping[self._provider]
                    red_url = item.assets[red_band].href
                    nir_url = item.assets[nir_band].href
                    green_url = item.assets[green_band].href
                    blue_url = item.assets[blue_band].href


                    with rasterio.Env(GDAL_CACHEMAX=512):
                        with rasterio.open(red_url) as red_src:
                            # Get window and transform
                            window = red_src.window(*bounds)
                            if not self.is_valid_window(window, red_src.width, red_src.height):
                                continue

                            window = self.round_window(window)
                            window_transform = red_src.window_transform(window)

                            # Read data
                            red = red_src.read(1, window=window).astype(float)
                            with rasterio.open(nir_url) as nir_src:
                                nir = nir_src.read(1, window=window).astype(float)

                            blue = None
                            if blue_url:
                                with rasterio.open(blue_url) as blue_src:
                                    blue = blue_src.read(1, window=window).astype(float)

                            green = None
                            if green_url:
                                with rasterio.open(green_url) as green_src:
                                    green = green_src.read(1, window=window).astype(float)

                            if red.size == 0 or nir.size == 0:
                                continue

                            if red.shape != nir.shape:
                                logger.warning(
                                    f"Shape mismatch for field {feature['GEO_ID']}: red={red.shape}, nir={nir.shape}")
                                continue
                            # Create mask with explicit transform
                            shapes = [geom.__geo_interface__]
                            mask = geometry_mask(
                                shapes,
                                out_shape=(window.height, window.width),
                                transform=window_transform,
                                invert=True
                            )
                            # Validate mask shape
                            if mask.shape != red.shape:
                                logger.warning(
                                    f"Mask shape mismatch for field {feature['GEO_ID']}: mask={mask.shape}, data={red.shape}")
                                continue
                            # Calculate indices only for valid pixels
                            valid_pixels = mask & (red > -1) & (nir > -1)
                            if not np.any(valid_pixels):
                                continue


                            # Initialize arrays for indices
                            ndvi = np.zeros_like(red)
                            evi = np.zeros_like(red)
                            savi = np.zeros_like(red)
                            msavi = np.zeros_like(red)
                            ndwi = np.zeros_like(red)
                            gci = np.zeros_like(red)

                            # Ensure all additional bands match shape before processing
                            if blue is not None and blue.shape != red.shape:
                                logger.warning(f"Blue band shape mismatch for field {feature['GEO_ID']}")
                                blue = None

                            if green is not None and green.shape != red.shape:
                                logger.warning(f"Green band shape mismatch for field {feature['GEO_ID']}")
                                green = None

                            # Calculate NDVI
                            ndvi[valid_pixels] = (nir[valid_pixels] - red[valid_pixels]) / (
                                    nir[valid_pixels] + red[valid_pixels])
                            ndvi = np.clip(ndvi, -1, 1)
                            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

                            # Calculate EVI if blue band is available
                            if blue is not None:
                                print(blue)
                                valid_pixels_evi = valid_pixels & (blue > -1)
                                evi[valid_pixels_evi] = 2.5 * (
                                        (nir[valid_pixels_evi] - red[valid_pixels_evi]) /
                                        (nir[valid_pixels_evi] + 6 * red[valid_pixels_evi] -
                                         7.5 * blue[valid_pixels_evi] + 1)
                                )
                                evi = np.nan_to_num(evi, nan=0.0, posinf=1.0, neginf=-1.0)
                                #evi = np.clip(evi, -1, 1)

                            # Calculate SAVI
                            L = 0.5  # soil brightness correction factor
                            savi[valid_pixels] = (
                                                         (nir[valid_pixels] - red[valid_pixels]) /
                                                         (nir[valid_pixels] + red[valid_pixels] + L)
                                                 ) * (1 + L)
                            savi = np.clip(savi, -1, 1)

                            # Calculate MSAVI
                            msavi[valid_pixels] = (
                                                          2 * nir[valid_pixels] + 1 -
                                                          np.sqrt((2 * nir[valid_pixels] + 1) ** 2 -
                                                                  8 * (nir[valid_pixels] - red[valid_pixels]))
                                                  ) / 2
                            msavi = np.nan_to_num(msavi, nan=0.0, posinf=1.0, neginf=-1.0)

                            # Calculate NDWI if green band is available
                            if green is not None:
                                valid_pixels_ndwi = valid_pixels & (green > -1)
                                ndwi[valid_pixels_ndwi] = (
                                        (green[valid_pixels_ndwi] - nir[valid_pixels_ndwi]) /
                                        (green[valid_pixels_ndwi] + nir[valid_pixels_ndwi] + 1e-10)
                                )
                                ndwi = np.clip(ndwi, -1, 1)

                                # calculate gci
                                valid_pixels_gci = valid_pixels & (green > -1)
                                gci[valid_pixels_gci] = (nir[valid_pixels_gci] / green[valid_pixels_gci]) - 1
                                gci = np.nan_to_num(gci, nan=0.0, posinf=1.0, neginf=-1.0)

                            # Store results
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
                    print(f"Error processing image for field {feature['GEO_ID']}: {str(e)}")
                    continue

            # Calculate statistics
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

                        if len(values) > 1:
                            days = [(date - results['dates'][0]).days for date in results['dates']]
                            try:
                                stats[f'{index_name}_trend'] = np.polyfit(days, values, 1)[0]
                            except:
                                stats[f'{index_name}_trend'] = None

                            peak_idx = np.argmax(values)
                            stats[f'{index_name}_time_to_peak'] = days[peak_idx]

                            changes = np.diff(values)
                            stats[f'{index_name}_max_increase'] = np.max(changes)
                            stats[f'{index_name}_max_decrease'] = np.min(changes)
                            stats[f'{index_name}_mean_change'] = np.mean(changes)

                            stats[f'{index_name}_cv'] = stats[f'{index_name}_std'] / stats[f'{index_name}_mean']

                        if  len(values) >= 3:  # These metrics need at least 3 points
                            stats[f'{index_name}_skewness'] = float(skew(values))
                            stats[f'{index_name}_kurtosis'] = float(kurtosis(values))
                        else:
                            logger.error(f"Error calculating skewness/kurtosis for field {feature['GEO_ID']}: {str(e)}")
                            logger.error(f"Values: {values}")
                            stats[f'{index_name}_skewness'] = 0
                            stats[f'{index_name}_kurtosis'] = 0

                        if  len(results['dates']) >= 3:  # Need at least 3 points for season divisions
                            stats[f'{index_name}_growing_season'] = days[-1]
                            stats[f'{index_name}_early_season_mean'] = np.mean(values[:len(values) // 3])
                            stats[f'{index_name}_mid_season_mean'] = np.mean(
                                values[len(values) // 3:2 * len(values) // 3])
                            stats[f'{index_name}_late_season_mean'] = np.mean(values[2 * len(values) // 3:])

                stats['n_observations'] = len(results['dates'])
                return stats
        except Exception as e:
            logger.error(f"Error processing field {feature['GEO_ID']}: {str(e)}")
            return None

    def is_valid_window(self, window, width, height):
        """Check if window is valid"""
        return not (window.col_off >= width or
                    window.row_off >= height or
                    window.col_off + window.width <= 0 or
                    window.row_off + window.height <= 0)

    def round_window(self, window):
        """Round window coordinates to integers"""
        return rasterio.windows.Window(
            col_off=int(window.col_off),
            row_off=int(window.row_off),
            width=int(window.width),
            height=int(window.height)
        )

    def save_features(self):
        """
        Save extracted features to CSV file.
        """
        try:
            logger.info(f"Total fields processed for CSV: {len(self.features)}")
            if self.features:
                # convert features dictionary to DataFrame
                features_df = pd.DataFrame.from_dict(self.features, orient='index')

                # create output directory if it doesn't exist
                output_dir = Path(r"/src/crop_classification")
                output_dir.mkdir(parents=True, exist_ok=True)

                # create output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f'training_features_{timestamp}.csv'

                # save to CSV
                features_df.to_csv(output_path, index=False)
                logger.info(f"Saved features to {output_path}")
            else:
                logger.warning(f"No features!")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise
